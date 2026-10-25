#!/usr/bin/env python3
"""Reproducible, retrieval-gated ADQA evaluation.

Replacement for the pasted notebook.  It preserves the JSON dataset formats and
the TSV KG workflow, but removes answer-leaking shortcuts and makes every
retrieval and decoding decision inspectable.  It intentionally does not promise
any target accuracy: run the ablations first, tune only on a development split,
and report the untouched test result once.

Required files (by default in --data-dir): medqa_filter.json, alz.txt,
MedQuAD.csv.  entity_embeddings.pkl and keyword_embeddings.pkl are optional.
Secrets: set HF_TOKEN (and, only when --enable-web is used,
FIRECRAWL_API_KEY) in the environment or Kaggle Secrets.  No secret is stored
in this file.

MODIFICATION 1 (G-Retriever only): --gretriever-embedding-mode {both,gnn,llm}
  both : GNN soft-prefix embedding + Llama-2 text embedding of the subgraph (original behaviour)
  gnn  : GNN soft-prefix embedding only (no triples / keywords given as text)
  llm  : Llama-2 text embedding of the serialized subgraph only (no GNN, no checkpoint)

MODIFICATION 2 (NEW, final answer stage only): permutation-debiased answer scoring
  --answer-mode perm         : the option order is cyclically shifted --perm-k times; the
                               exact log-probability of each answer letter is mapped back to
                               the original option and averaged (removes position bias and
                               lowers variance).  Uses the evidence when there is some.
  --answer-mode perm_fusion  : same, but the evidence-free and the evidence-conditioned
                               permutation ensembles are fused log-linearly:
                                 final = (1-w) * no_evidence + w * with_evidence
                               (w = --perm-evidence-weight).  This protects against
                               retrieved text that is related but misleading.
  All retrieval components (SPARQL KG, OG-RAG, OLS, G-Retriever, gates) are unchanged.
  The HF token is now read from the HF_TOKEN environment variable only.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, re, time
import warnings
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from rdflib import Graph, Namespace, URIRef
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = "ABCDE"


def normalise(text: str) -> str:
    text = text.lower()
    # Frequent machine-translation variants in this MedQA subset.  This makes
    # ontology matching robust; it does not read or infer gold labels.
    aliases = {"ezihaimo": "alzheimer", "azihaimo": "alzheimer",
               "alzheimer s": "alzheimer", "alzheimers": "alzheimer",
               "winnick": "wernicke", "cherrie": "chemotherapy",
               "rosen saybolt": "rosenthal", "stellate cell": "astrocyte"}
    for source, target in aliases.items():
        text = re.sub(r"\b" + re.escape(source) + r"\b", target, text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text)).strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalise(text))


def option_letter(text: str, valid: Iterable[str]) -> str:
    valid = {x.upper() for x in valid}
    m = re.search(r"(?:answer|option|letter|choice)?\s*[:=\-]?\s*[\[(]?([A-Ea-e])(?:[\])\s.,;:]|$)", str(text))
    return m.group(1).upper() if m and m.group(1).upper() in valid else ""


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def hf_token() -> str | None:
    """Token comes only from the environment (never hard-coded)."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None


@dataclass
class Config:
    data_dir: Path
    output_dir: Path
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    main_device: str = "auto"
    embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    seed: int = 42
    max_input_tokens: int = 8192
    retrieval_top_k: int = 3
    retrieval_pool: int = 30
    kg_top_k: int = 6
    # Direct KG context can be retrieved either by lexical grounding (legacy)
    # or by the SPARQL path-and-neighbour construction below.
    kg_context_mode: str = "sparql"  # direct | sparql
    sparql_entity_seeds: int = 2
    sparql_min_similarity: float = 0.35
    sparql_max_hops: int = 3
    sparql_max_paths: int = 4
    sparql_neighbor_limit: int = 12
    sparql_neighbor_expand_k: int = 3
    sparql_rerank_top_k: int = 6
    sparql_mmr_lambda: float = 0.75
    max_evidence_chars: int = 5000
    # Retrieval is opt-in: the measured ablation on the supplied small KG showed
    # that both KG and MedQuAD reduced accuracy.  Re-enable a source only if its
    # development-set ablation is positive.
    use_kg: bool = False
    use_medquad: bool = False
    use_ograg: bool = False
    online_ontologies: bool = False
    ols_timeout: int = 45
    ograg_force_kg: bool = False
    ograg_evidence_router: bool = False
    ograg_router_margin: float = 0.35
    ograg_safe_fallback: bool = False
    ograg_abstain_without_kg: bool = False
    ograg_append_ontology_anchor: bool = False
    # MedQuAD reference text is not an online ontology annotation. Keep this
    # source opt-in so the OLS ablation measures OLS rather than mixed OLS and
    # MedQuAD context.
    ograg_append_medquad_anchor: bool = False
    ontology_anchor_chars: int = 360
    ograg_append_hyperedges: bool = False
    ograg_hyperedge_chars: int = 700
    # Optional evidence gate for supplementary blocks.  Its tolerance must be
    # selected on the development split and frozen before a test evaluation.
    gate_supplements: bool = False
    supplement_score_tolerance: float = 0.25
    # Source-specific overrides: a negative tolerance is stricter, requiring
    # the supplement to outrank direct KG evidence by its absolute value.
    # None keeps the shared tolerance for backwards-compatible ablations.
    ols_supplement_tolerance: float | None = None
    gretriever_supplement_tolerance: float | None = None
    ols_seed_kg_entities: bool = True
    ols_fetch_neighbors: bool = False
    ols_neighbor_max: int = 2
    use_gretriever: bool = False
    # Native G-Retriever graph encoder.  This mode is intentionally opt-in and
    # requires a trained checkpoint for the GAT and projector; random GNN
    # weights are not valid experimental evidence.
    use_gretriever_gnn: bool = False
    gretriever_untrained_gnn: bool = False
    gretriever_gnn_checkpoint: Path | None = None
    gretriever_gnn_hidden_dim: int = 384
    gretriever_gnn_heads: int = 4
    gretriever_soft_tokens: int = 8
    # Which subgraph embedding(s) are given to Llama-2: both | gnn | llm
    gretriever_embedding_mode: str = "llm"
    gretriever_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    gretriever_device: str = "cpu"
    gretriever_topk_nodes: int = 50
    gretriever_topk_edges: int = 3
    gretriever_entity_seeds: int = 8
    gretriever_entity_min_similarity: float = 0.35
    gretriever_max_new_tokens: int = 96
    gretriever_max_triples: int = 30
    gretriever_paragraph_chars: int = 700
    gretriever_require_direct_kg: bool = True
    use_gretriever_subgraph: bool = False
    gretriever_max_nodes: int = 12
    gretriever_max_edges: int = 12
    gretriever_chars: int = 1000
    # Optional post-PCST evidence ranking. It only orders the triples already
    # selected by PCST with the question text; answer options and gold labels
    # are excluded from this stage.
    gretriever_rerank_triples: bool = False
    gretriever_rerank_top_k: int = 6
    enable_web: bool = False
    # A negative cross-encoder score is not useful evidence. Tune this only on dev.
    min_reranker_score: float = 0.0
    dataset: str = "medqa"
    limit: int | None = None
    split: str = "all"                 # all | dev | test, deterministic by id
    dev_fraction: float = 0.15
    ablations: bool = False
    answer_mode: str = "letter"        # letter | cot_letter | binary | perm | perm_fusion
    cot_max_new_tokens: int = 180
    # NEW: permutation-ensemble settings (answer_mode perm / perm_fusion).
    perm_k: int = 4                    # number of cyclic option shifts (<= number of options)
    perm_evidence_weight: float = 0.5  # w in (1-w)*no_evidence + w*with_evidence (perm_fusion)
    # Few-shot demonstrations are selected only from a disjoint labelled pool.
    # They are never retrieved from the evaluated split.
    fewshot_k: int = 0
    fewshot_pool_split: str = "dev"    # dev | test; must differ from --split
    fewshot_pool_file: Path | None = None
    fewshot_strategy: str = "semantic" # semantic | random
    fewshot_max_chars: int = 1800
    # Never use gold labels for retrieval/prompt construction. Labels are evaluation only.
    save_every: int = 1


class Processor:
    IS_EXTRACTIVE = False
    def __init__(self, cfg: Config, filename: str):
        raw = json.loads((cfg.data_dir / filename).read_text(encoding="utf-8"))
        # Optional NER files may contain stale/hand-created fields.  Keep only fields
        # from the raw benchmark so no NER artifact can accidentally include a label.
        self.data_original = raw
        self.data = raw

    @staticmethod
    def render(question: str, choices: list[str]) -> str:
        return question.strip() + "\n\n" + "\n".join(
            f"{LETTERS[i]}. {choice.strip()}" for i, choice in enumerate(choices))


class MedQAProcessor(Processor):
    def __init__(self, cfg: Config): super().__init__(cfg, "medqa_filter.json")
    def question_choices(self, x): return str(x["question"]), list(x["choices"])
    def gold(self, x): return LETTERS[list(x["choices"]).index(x["answer"][0])]


class MedMCQAProcessor(Processor):
    def __init__(self, cfg: Config): super().__init__(cfg, "medmcqa_filter.json")
    def question_choices(self, x): return str(x["question"]), [x["opa"], x["opb"], x["opc"], x["opd"]]
    def gold(self, x): return LETTERS[int(x["cop"])]


class MMLUProcessor(Processor):
    IS_EXTRACTIVE = True
    def __init__(self, cfg: Config): super().__init__(cfg, "mmlu_filter.json")
    def question_choices(self, x): return str(x["question"]), list(x["choices"])
    def gold(self, x): return LETTERS[int(x["answer"])]


class QA4MREProcessor(Processor):
    IS_EXTRACTIVE = True
    def __init__(self, cfg: Config): super().__init__(cfg, "qa4mre_filter.json")
    def question_choices(self, x): return str(x["question_str"]), list(x["answer_options"]["answer_str"])
    def gold(self, x): return LETTERS[int(x["correct_answer_id"]) - 1]


PROCESSORS = {"medqa": MedQAProcessor, "medmcqa": MedMCQAProcessor,
              "mmlu": MMLUProcessor, "qa4mre": QA4MREProcessor}


class LocalModel:
    """Scores complete answer letters, instead of sampling a chain-of-thought.

    The prior code used the first token-id of ' A'/'A'; that is not necessarily
    the complete letter token and samples three noisy rationales. Here all legal
    letters are scored by their exact conditional log-likelihood, deterministically.
    """
    def __init__(self, cfg: Config):
        token = ""
        kwargs = {"token": token} if token else {}
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, **kwargs)
        if cfg.main_device != "auto" and cfg.main_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("--main-device requests CUDA but CUDA is unavailable")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else (torch.float16 if torch.cuda.is_available() else torch.float32)
        device_map: str | dict[str, str] = "auto" if cfg.main_device == "auto" else {"": cfg.main_device}
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True, **kwargs).eval()
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        self.max_tokens = cfg.max_input_tokens
        self._letter_ids: dict[int, list[int]] = {}

    def chat_prefix(self, messages: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        """Deterministic private reasoning.  Its final answer is never parsed here."""
        text = self.chat_prefix(messages)
        encoded = self.tok(text, return_tensors="pt", add_special_tokens=False,
                           truncation=True, max_length=self.max_tokens)
        encoded = {k: v.to(next(self.model.parameters()).device) for k, v in encoded.items()}
        output = self.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def score_completions(self, prefix: str, completions: list[str]) -> dict[str, float]:
        # Common prefix tokenization must be exact; score every token in each candidate.
        enc_prefix = self.tok(prefix, add_special_tokens=False).input_ids
        out: dict[str, float] = {}
        device = next(self.model.parameters()).device
        for completion in completions:
            full = self.tok(prefix + completion, add_special_tokens=False).input_ids
            target = full[len(enc_prefix):]
            if not target: out[completion] = float("-inf"); continue
            full = full[-self.max_tokens:]
            # If truncation removed prefix, recompute target boundary conservatively.
            start = max(0, len(full) - len(target))
            ids = torch.tensor([full], device=device)
            logits = self.model(input_ids=ids).logits[0]
            logp = torch.log_softmax(logits.float(), dim=-1)
            score = sum(float(logp[start + j - 1, tok_id]) for j, tok_id in enumerate(target) if start + j - 1 >= 0)
            out[completion] = score / max(1, len(target))
        return out

    def _ids_for_letter(self, index: int) -> list[int]:
        """Token ids that spell option letter `index` (with and without a leading space)."""
        if index not in self._letter_ids:
            ids = set()
            for form in (LETTERS[index], " " + LETTERS[index]):
                encoded = self.tok(form, add_special_tokens=False).input_ids
                if encoded: ids.add(int(encoded[0]))
            self._letter_ids[index] = sorted(ids)
        return self._letter_ids[index]

    @torch.no_grad()
    def letter_logprobs(self, prefix: str, n_options: int) -> np.ndarray:
        """NEW: one forward pass -> log-probability of each answer letter, renormalised over
        the legal letters only.  Equivalent to the exact first-token scoring but n times cheaper."""
        ids = self.tok(prefix, add_special_tokens=False).input_ids[-self.max_tokens:]
        device = next(self.model.parameters()).device
        logits = self.model(input_ids=torch.tensor([ids], device=device)).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        raw = np.asarray([float(torch.logsumexp(logp[self._ids_for_letter(i)], dim=0))
                          for i in range(n_options)], dtype="float64")
        raw -= raw.max()
        return raw - np.log(np.exp(raw).sum())


class HybridIndex:
    def __init__(self, docs: list[dict[str, str]], embedder: SentenceTransformer):
        self.docs = docs
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokens(d["text"]) or ["_"] for d in docs])
        self.embeddings = embedder.encode([d["text"] for d in docs], normalize_embeddings=True,
                                          convert_to_numpy=True, show_progress_bar=False).astype("float32")

    def search(self, query: str, reranker: CrossEncoder, pool: int, top_k: int, minimum: float) -> list[dict[str, Any]]:
        if not self.docs: return []
        qtok = tokens(query) or ["_"]
        bm = self.bm25.get_scores(qtok)
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        dense = self.embeddings @ q
        # Rank fusion avoids incompatible BM25/dense score scales.
        candidates = sorted(range(len(self.docs)), key=lambda i: (-self._rrf_rank(bm, i) - self._rrf_rank(dense, i)))[:min(pool, len(self.docs))]
        pairs = [(query, self.docs[i]["text"]) for i in candidates]
        ce = reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, ce), key=lambda x: float(x[1]), reverse=True)
        return [{**self.docs[i], "reranker_score": float(s)} for i, s in ranked[:top_k] if float(s) >= minimum]

    @staticmethod
    def _rrf_rank(scores: np.ndarray, index: int, k: int = 60) -> float:
        # Invert a rank without retaining a large additional sorted list in the index.
        rank = int(np.sum(scores > scores[index]))
        return 1.0 / (k + rank + 1)


def load_medquad(path: Path) -> list[dict[str, str]]:
    """Use the actual MedQuAD schema, not Name/Symptoms/Treatments.

    Each document retains focus/question/answer and source metadata for auditing.
    """
    docs, seen = [], set()
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            answer = (row.get("answer") or "").strip()
            if not answer: continue
            focus, question = (row.get("question_focus") or "").strip(), (row.get("question") or "").strip()
            text = f"Topic: {focus}\nReference question: {question}\nReference answer: {answer}"
            key = normalise(text)
            if key not in seen:
                seen.add(key); docs.append({"id": row.get("question_id", ""), "source": row.get("document_source", ""), "text": text})
    return docs


class KnowledgeGraph:
    def __init__(self, path: Path):
        self.edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.labels: list[str] = []
        if not path.exists(): return
        labels = set()
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) < 3 or not all(x.strip() for x in row[:3]): continue
                h, rel, t = (x.strip() for x in row[:3])
                self.edges[h].append((rel, t)); self.edges[t].append((f"inverse:{rel}", h)); labels.update((h, t))
        self.labels = sorted(labels)

    def evidence(self, question: str, options: list[str], max_paths: int) -> list[str]:
        # A graph node must occur in the vignette itself.  Merely occurring in a
        # distractor must never activate a KG path.
        text = normalise(question)
        seeds = [x for x in self.labels if len(normalise(x)) >= 3 and normalise(x) in text]
        seeds = sorted(seeds, key=lambda x: -len(normalise(x)))[:8]
        # The non-seed side of a retained fact must be relevant to a clinical
        # finding or to an answer option.  This prevents one generic Alzheimer
        # node from injecting all its unrelated facts into every question.
        support_terms = {w for w in tokens(question + " " + " ".join(options)) if len(w) >= 4}
        facts: list[str] = []
        for seed in seeds:
            for rel, other in self.edges[seed][:max_paths]:
                fact_terms = {w for w in tokens(rel + " " + other) if len(w) >= 4}
                if fact_terms & support_terms:
                    facts.append(f"{seed} --[{rel}]--> {other}")
        return facts[:max_paths]


class SparqlKGContext:
    """Question-seeded SPARQL paths plus relevance-filtered neighbourhoods.

    The AD-KG is converted once to an in-memory RDF graph. Query keywords are
    aligned to AD-KG labels through cosine similarity; the two best distinct
    KG entities become seeds. SPARQL retrieves paths of at most three hops
    between them and outgoing local edges (including materialized inverse
    edges). Relevant first-hop neighbours are expanded once. Finally, a
    cross-encoder and maximal marginal relevance (MMR) retain a compact,
    diverse list of factual triples. The answer options and gold label are not
    used anywhere in this construction or reranking stage.
    """
    PREFIX = Namespace("urn:adkg:")

    def __init__(self, kg: KnowledgeGraph, embedder: SentenceTransformer):
        self.kg, self.embedder = kg, embedder
        self.graph = Graph()
        self.node_uri: dict[str, URIRef] = {}
        self.uri_node: dict[URIRef, str] = {}
        self.rel_uri: dict[str, URIRef] = {}
        self.uri_rel: dict[URIRef, str] = {}
        self.labels = list(kg.labels)
        self.node_vectors = embedder.encode(
            self.labels or ["empty"], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False).astype("float32")
        self.label_index = {label: i for i, label in enumerate(self.labels)}
        seen = set()
        for head, links in kg.edges.items():
            for relation, tail in links:
                if relation.startswith("inverse:"):
                    continue
                triple = (head, relation, tail)
                if triple in seen:
                    continue
                seen.add(triple)
                self.graph.add((self._node(head), self._relation(relation), self._node(tail)))
                # Materialized inverse edges let the bounded SPARQL queries
                # explore an undirected local topology without unsafe string
                # interpolation of biomedical entity labels.
                self.graph.add((self._node(tail), self._relation("inverse:" + relation), self._node(head)))

    def _node(self, label: str) -> URIRef:
        if label not in self.node_uri:
            uri = URIRef(str(self.PREFIX) + "node/" + hashlib.sha256(label.encode("utf-8")).hexdigest())
            self.node_uri[label] = uri
            self.uri_node[uri] = label
        return self.node_uri[label]

    def _relation(self, label: str) -> URIRef:
        if label not in self.rel_uri:
            uri = URIRef(str(self.PREFIX) + "relation/" + hashlib.sha256(label.encode("utf-8")).hexdigest())
            self.rel_uri[label] = uri
            self.uri_rel[uri] = label
        return self.rel_uri[label]

    def _fact(self, source: URIRef, predicate: URIRef, target: URIRef) -> str:
        return f"{self.uri_node[source]} --[{self.uri_rel[predicate]}]--> {self.uri_node[target]}"

    def _query(self, body: str, bindings: dict[str, URIRef]) -> list[Any]:
        return list(self.graph.query("SELECT * WHERE { " + body + " }", initBindings=bindings))

    def _seed_entities(self, question: str, count: int, minimum: float) -> list[dict[str, Any]]:
        # Use normalized biomedical keywords extracted from the question only;
        # answer options and dataset metadata are intentionally excluded.
        keywords = question_keywords_for_kg(question)
        if not keywords:
            return []
        keyword_vectors = self.embedder.encode(keywords, normalize_embeddings=True,
                                                convert_to_numpy=True, show_progress_bar=False).astype("float32")
        similarities = self.node_vectors @ keyword_vectors.T
        best_keyword = np.argmax(similarities, axis=1)
        scores = similarities[np.arange(len(self.labels)), best_keyword]
        ranked = np.argsort(scores)[::-1]
        selected = []
        for idx in ranked:
            score = float(scores[int(idx)])
            if score < minimum:
                break
            selected.append({"entity": self.labels[int(idx)], "similarity": score,
                             "matched_question_keyword": keywords[int(best_keyword[int(idx)])]})
            if len(selected) >= count:
                break
        return selected

    def _paths(self, first: URIRef, second: URIRef, max_hops: int, maximum: int) -> list[list[str]]:
        paths: list[list[str]] = []
        # All paths use materialized inverse relations; a path may therefore
        # traverse either original direction while preserving its direction in
        # the emitted predicate label.
        if max_hops >= 1:
            for row in self._query("?s ?p1 ?t .", {"s": first, "t": second})[:maximum]:
                paths.append([self._fact(first, row.p1, second)])
        if max_hops >= 2 and len(paths) < maximum:
            for row in self._query("?s ?p1 ?m . ?m ?p2 ?t .", {"s": first, "t": second})[:maximum - len(paths)]:
                paths.append([self._fact(first, row.p1, row.m), self._fact(row.m, row.p2, second)])
        if max_hops >= 3 and len(paths) < maximum:
            for row in self._query("?s ?p1 ?m1 . ?m1 ?p2 ?m2 . ?m2 ?p3 ?t .", {"s": first, "t": second})[:maximum - len(paths)]:
                paths.append([self._fact(first, row.p1, row.m1), self._fact(row.m1, row.p2, row.m2), self._fact(row.m2, row.p3, second)])
        return paths

    def _neighbours(self, source: URIRef, limit: int) -> list[tuple[str, URIRef]]:
        rows = self._query("?s ?p ?n .", {"s": source})[:limit]
        return [(self._fact(source, row.p, row.n), row.n) for row in rows]

    @staticmethod
    def _mmr(facts: list[str], ce_scores: np.ndarray, q_vector: np.ndarray,
             embedder: SentenceTransformer, maximum: int, lam: float) -> tuple[list[str], list[float]]:
        if not facts:
            return [], []
        vectors = embedder.encode(facts, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype("float32")
        ce = np.asarray(ce_scores, dtype="float32")
        ce = (ce - ce.min()) / (ce.max() - ce.min() + 1e-8)
        selected: list[int] = []
        candidates = list(range(len(facts)))
        while candidates and len(selected) < min(maximum, len(facts)):
            best, best_score = -1, -float("inf")
            for i in candidates:
                novelty = max((float(vectors[i] @ vectors[j]) for j in selected), default=0.0)
                relevance = float(vectors[i] @ q_vector)
                score = lam * (0.65 * float(ce[i]) + 0.35 * relevance) - (1.0 - lam) * novelty
                if score > best_score:
                    best, best_score = i, score
            selected.append(best)
            candidates.remove(best)
        return [facts[i] for i in selected], [float(ce_scores[i]) for i in selected]

    def retrieve(self, question: str, reranker: CrossEncoder, cfg: Config) -> tuple[list[str], dict[str, Any]]:
        seeds = self._seed_entities(question, cfg.sparql_entity_seeds, cfg.sparql_min_similarity)
        if not seeds:
            return [], {"seed_entities": [], "paths": [], "candidate_triples": [], "selected_triples": []}
        seed_uris = [self._node(item["entity"]) for item in seeds]
        paths = self._paths(seed_uris[0], seed_uris[1], cfg.sparql_max_hops, cfg.sparql_max_paths) if len(seed_uris) >= 2 else []
        candidates: list[str] = [fact for path in paths for fact in path]
        first_hop: list[tuple[str, URIRef]] = []
        for uri in seed_uris:
            first_hop.extend(self._neighbours(uri, cfg.sparql_neighbor_limit))
        candidates.extend(fact for fact, _ in first_hop)
        # Expand only locally relevant first-hop neighbours, rather than every
        # neighbour, to keep the context compact and clinically focused.
        q_vector = self.embedder.encode([question], normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False)[0]
        ranked_neighbours = sorted(first_hop, key=lambda pair: float(self.node_vectors[self.label_index[self.uri_node[pair[1]]]] @ q_vector), reverse=True)
        expanded = []
        for _, uri in ranked_neighbours[:cfg.sparql_neighbor_expand_k]:
            expanded.extend(self._neighbours(uri, cfg.sparql_neighbor_limit))
        candidates.extend(fact for fact, _ in expanded)
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return [], {"seed_entities": seeds, "paths": paths, "candidate_triples": [], "selected_triples": []}
        ce_scores = reranker.predict([(question, fact) for fact in candidates], show_progress_bar=False)
        selected, selected_scores = self._mmr(candidates, ce_scores, q_vector, self.embedder,
                                              cfg.sparql_rerank_top_k, cfg.sparql_mmr_lambda)
        return selected, {
            "seed_entities": seeds,
            "paths": paths,
            "first_hop_count": len(first_hop),
            "expanded_neighbour_count": len(expanded),
            "candidate_triples": candidates,
            "selected_triples": selected,
            "cross_encoder_scores": selected_scores,
            "reranker": "cross-encoder + MMR",
        }


class GRetrieverSubgraph:
    """G-Retriever-style *retrieval* over alz.txt, rendered as evidence text.

    The supplied G-Retriever code uses an untrained GAT; its GNN vectors are
    random and cannot validly improve medical QA. This implementation preserves
    the useful retrieval component: semantic node/edge prizes plus PCST (with a
    deterministic BFS fallback), then passes an attributable subgraph paragraph
    to the existing LLM. It consumes `alz.txt` directly -- no JSON conversion
    files, GAT checkpoint, or GPU tensor placement are required.
    """
    def __init__(self, kg: KnowledgeGraph, embedder: SentenceTransformer):
        self.kg, self.embedder = kg, embedder
        self.nodes = list(kg.labels)
        self.node_to_id = {n: i for i, n in enumerate(self.nodes)}
        seen, triples = set(), []
        for head, links in kg.edges.items():
            for rel, tail in links:
                if rel.startswith("inverse:") or (head, rel, tail) in seen: continue
                seen.add((head, rel, tail)); triples.append((head, rel, tail))
        self.triples = triples
        self.edge_pairs = np.asarray([[self.node_to_id[h], self.node_to_id[t]] for h, _, t in triples], dtype=np.int64) if triples else np.empty((0, 2), dtype=np.int64)
        self.node_vectors = embedder.encode(self.nodes or ["empty"], normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False).astype("float32")
        self.relation_vectors = embedder.encode([r for _, r, _ in triples] or ["empty"], normalize_embeddings=True,
                                                convert_to_numpy=True, show_progress_bar=False).astype("float32")

    def _pcst(self, node_prizes: np.ndarray, edge_prizes: np.ndarray, max_nodes: int) -> tuple[set[int], set[int]]:
        if not len(self.edge_pairs): return set(np.argsort(node_prizes)[-max_nodes:]), set()
        try:
            from pcst_fast import pcst_fast
            # High semantic edge relevance means a low PCST traversal cost.
            costs = np.maximum(0.01, 1.0 - edge_prizes)
            vertices, selected_edges = pcst_fast(self.edge_pairs, node_prizes, costs,
                                                  -1, 1, 'gw', 0)
            nodes = set(map(int, vertices)); edges = set(map(int, selected_edges))
            if nodes: return set(sorted(nodes, key=lambda i: node_prizes[i], reverse=True)[:max_nodes]), edges
        except Exception:
            pass
        # Dependency-free deterministic fallback: top semantic nodes and their
        # highest-prize incident edges. It is still a genuine query subgraph.
        nodes = set(map(int, np.argsort(node_prizes)[-max_nodes:]))
        edges = {i for i, (u, v) in enumerate(self.edge_pairs) if int(u) in nodes or int(v) in nodes}
        return nodes, edges

    def match_question_keywords_to_entities(self, question_keywords: list[str],
                                            maximum: int, minimum_similarity: float) -> list[dict[str, Any]]:
        """Map question keywords to AD-KG entities through cosine similarity.

        Node and keyword embeddings are L2-normalized, therefore the matrix
        product is cosine similarity.  Only the resulting KG entity names, not
        the question keywords themselves, are passed to PCST.
        """
        if not self.nodes or not question_keywords:
            return []
        keyword_vectors = self.embedder.encode(question_keywords,
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False).astype("float32")
        similarity = self.node_vectors @ keyword_vectors.T
        best_keyword_index = np.argmax(similarity, axis=1)
        best_scores = similarity[np.arange(len(self.nodes)), best_keyword_index]
        ranked = np.argsort(best_scores)[::-1]
        matches = []
        for node_id in ranked:
            score = float(best_scores[node_id])
            if score < minimum_similarity:
                break
            matches.append({"entity": self.nodes[int(node_id)], "score": score,
                            "matched_question_keyword": question_keywords[int(best_keyword_index[node_id])]})
            if len(matches) >= maximum:
                break
        return matches

    def retrieve(self, keywords: list[str], max_nodes: int, max_edges: int,
                 max_chars: int) -> tuple[str, dict[str, Any]]:
        if not self.triples: return "", {"nodes": 0, "edges": 0, "used_pcst": False}
        # Deliberately do not put the benchmark question into PCST. The graph is
        # selected from canonical KG/OLS terms only, as requested.
        keywords = list(dict.fromkeys(x.strip() for x in keywords if x and x.strip()))[:16]
        if not keywords:
            return "", {"nodes": 0, "edges": 0, "used_pcst": False, "keywords": []}
        query = "Ontology/KG concepts: " + ", ".join(keywords)
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        node_prizes = np.maximum(0.0, self.node_vectors @ q)
        edge_prizes = np.maximum(0.0, self.relation_vectors @ q)
        selected_nodes, selected_edges = self._pcst(node_prizes, edge_prizes, max_nodes)
        # Ensure edges have both endpoints in the displayed subgraph, then rank.
        candidate_edges = [i for i in selected_edges if int(self.edge_pairs[i, 0]) in selected_nodes and int(self.edge_pairs[i, 1]) in selected_nodes]
        if not candidate_edges:
            candidate_edges = [i for i, (u, v) in enumerate(self.edge_pairs) if int(u) in selected_nodes or int(v) in selected_nodes]
        candidate_edges = sorted(candidate_edges, key=lambda i: float(edge_prizes[i] + node_prizes[self.edge_pairs[i, 0]] + node_prizes[self.edge_pairs[i, 1]]), reverse=True)[:max_edges]
        lines = [f"{self.triples[i][0]} --[{self.triples[i][1]}]--> {self.triples[i][2]}" for i in candidate_edges]
        paragraph = "[G-Retriever selected subgraph]\n" + "\n".join(lines)
        return paragraph[:max_chars], {"nodes": len(selected_nodes), "edges": len(candidate_edges),
                                        "keywords": keywords, "triples": lines,
                                        "node_ids": sorted(map(int, selected_nodes)),
                                        "edge_ids": list(map(int, candidate_edges)),
                                        "used_pcst": bool(selected_edges)}


class KeywordOnlyGRetrieverLLM:
    """Second LLM stage: Llama-2 verbalizes an already selected PCST graph.

    This is the textual-generation part of a G-Retriever pipeline. It receives
    only canonical OLS/KG keywords and selected KG triples--never the MCQ text,
    options, gold label, or answer. Llama-3 receives its resulting paragraph
    together with the usual OG-RAG/KG/OLS evidence and makes the final choice.
    """
    def __init__(self, cfg: Config):
        token = ""
        kwargs = {"token": token} if token else {}
        self.tok = AutoTokenizer.from_pretrained(cfg.gretriever_model_name, **kwargs)
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        requested = cfg.gretriever_device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("--gretriever-device requests CUDA but CUDA is unavailable")
        # CPU is the safe default: Llama-3 can retain the GPU(s), and no tensors
        # are ever shared between these two models.
        dtype = torch.float32 if requested == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.gretriever_model_name, torch_dtype=dtype,
            low_cpu_mem_usage=True, **kwargs).eval().to(torch.device(requested))
        self.device = torch.device(requested)
        self.max_input_tokens = 4096

    @torch.no_grad()
    def paragraph(self, keywords: list[str], triples: list[str], max_new_tokens: int) -> str:
        prompt = (
            "<s>[INST] You are the graph-to-text component of a medical retrieval system. "
            "Write one concise factual paragraph using ONLY the supplied graph triples. "
            "Do not answer a medical question, do not select an option, do not add facts, "
            "and say that information is unavailable when triples are insufficient.\n\n"
            "Canonical ontology/KG keywords:\n- " + "\n- ".join(keywords) +
            "\n\nSelected PCST graph triples:\n- " + "\n- ".join(triples) + " [/INST]"
        )
        encoded = self.tok(prompt, return_tensors="pt", truncation=True,
                           max_length=self.max_input_tokens).to(self.device)
        generated = self.model.generate(**encoded, do_sample=False,
            max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id)
        return self.tok.decode(generated[0, encoded["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()


class PureTorchGraphAttention(nn.Module):
    """Small dependency-free multi-head graph-attention layer.

    It is used only for the explicit untrained-GNN ablation when PyG is absent.
    The selected PCST graphs contain at most a few dozen nodes, so its dense
    attention calculation is intentionally simple and adequate at this scale.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.heads, self.out_channels = heads, out_channels
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_source = nn.Parameter(torch.empty(heads, out_channels))
        self.att_target = nn.Parameter(torch.empty(heads, out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_source)
        nn.init.xavier_uniform_(self.att_target)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        count = x.shape[0]
        h = self.linear(x).view(count, self.heads, self.out_channels)
        # mask[source, target]: retain PCST edges and every self-loop.
        mask = torch.eye(count, dtype=torch.bool, device=x.device)
        if edge_index.numel():
            mask[edge_index[0], edge_index[1]] = True
        source_score = (h * self.att_source).sum(dim=-1)  # [node, head]
        target_score = (h * self.att_target).sum(dim=-1)
        logits = torch.nn.functional.leaky_relu(
            source_score.T[:, :, None] + target_score.T[:, None, :], negative_slope=0.2)
        logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        # For every target, normalize relevance across its incoming sources.
        weights = torch.softmax(logits, dim=1)
        output = torch.einsum("hij,ihc->jhc", weights, h).reshape(count, -1)
        return output + self.bias


class TrainedGNNGraphToTextLLM:
    """Native G-Retriever-style GAT--MLP soft-prefix graph textualizer.

    PCST remains responsible for selecting a small keyword-grounded graph.  A
    *trained* GAT encodes that graph, and a trained MLP turns the graph vector
    into a short soft prefix for Llama-2.  The model sees only selected triples
    and canonical KG/OLS keywords; it never sees the benchmark MCQ, choices or
    label.  The final Llama-3 model remains the only answer selector.

    EMBEDDING MODES (cfg.gretriever_embedding_mode):
      both : [GNN soft prefix] + [Llama-2 token embeddings of keywords+triples]
      gnn  : [GNN soft prefix] only; triples/keywords are NOT given as text
      llm  : [Llama-2 token embeddings of keywords+triples] only; no GNN and no
             checkpoint are needed
    """
    def __init__(self, cfg: Config, subgraph: GRetrieverSubgraph):
        self.mode = cfg.gretriever_embedding_mode
        if self.mode not in ("both", "gnn", "llm"):
            raise ValueError("--gretriever-embedding-mode must be one of: both, gnn, llm")
        self.use_gnn = self.mode in ("both", "gnn")
        self.use_text = self.mode in ("both", "llm")
        if self.use_gnn:
            if not cfg.gretriever_gnn_checkpoint and not cfg.gretriever_untrained_gnn:
                raise ValueError(
                    "--use-gretriever-gnn requires --gretriever-gnn-checkpoint, or explicitly "
                    "pass --gretriever-untrained-gnn for a reproducible random-weight ablation.")
            if cfg.gretriever_gnn_checkpoint and not cfg.gretriever_gnn_checkpoint.is_file():
                raise FileNotFoundError(cfg.gretriever_gnn_checkpoint)
            try:
                from torch_geometric.nn import GATv2Conv, global_mean_pool
                self.GATv2Conv, self.global_mean_pool = GATv2Conv, global_mean_pool
                self.graph_backend = "torch_geometric_gatv2"
            except ImportError:
                if cfg.gretriever_gnn_checkpoint:
                    raise RuntimeError(
                        "This trained GNN checkpoint requires torch-geometric. Install it with "
                        "python3 -m pip install torch_geometric.")
                # A random-weight ablation does not need PyG. Use the local,
                # dependency-free attention implementation instead.
                self.GATv2Conv = PureTorchGraphAttention
                self.global_mean_pool = lambda features, batch: features.mean(dim=0, keepdim=True)
                self.graph_backend = "pure_torch_attention_untrained"
                warnings.warn("torch_geometric is unavailable; using the built-in PyTorch graph-attention "
                              "fallback for the untrained-GNN ablation.", stacklevel=2)
        else:
            self.graph_backend = "none_llm_text_only"
        self.cfg, self.subgraph = cfg, subgraph
        requested = cfg.gretriever_device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("--gretriever-device requests CUDA but CUDA is unavailable")
        if requested.startswith("cuda:"):
            try:
                ordinal = int(requested.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError("--gretriever-device must be cpu, cuda:0, cuda:1, etc.") from exc
            visible = torch.cuda.device_count()
            if ordinal < 0 or ordinal >= visible:
                raise RuntimeError(
                    f"--gretriever-device={requested} is invalid: this process sees {visible} CUDA device(s) "
                    f"numbered cuda:0 to cuda:{visible - 1}. Use --gretriever-device cpu, choose a visible "
                    "device, or expose both GPUs through CUDA_VISIBLE_DEVICES before launching Python.")
        self.device = torch.device(requested)
        token = ""
        kwargs = {"token": token} if token else {}
        self.tok = AutoTokenizer.from_pretrained(cfg.gretriever_model_name, **kwargs)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float32 if requested == "cpu" else torch.float16
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.gretriever_model_name, torch_dtype=dtype,
            low_cpu_mem_usage=True, **kwargs).eval().to(self.device)
        for parameter in self.llm.parameters():
            parameter.requires_grad_(False)
        self.gnn = None
        self.projector = None
        if self.use_gnn:
            input_dim = int(subgraph.node_vectors.shape[1])
            hidden = cfg.gretriever_gnn_hidden_dim
            heads = cfg.gretriever_gnn_heads
            if hidden % heads:
                raise ValueError("--gretriever-gnn-hidden-dim must be divisible by --gretriever-gnn-heads.")
            self.gnn = nn.ModuleDict({
                "gat1": self._make_gat(input_dim, hidden // heads, heads),
                "gat2": self._make_gat(hidden, hidden // heads, heads),
            }).to(self.device)
            llm_dim = int(self.llm.config.hidden_size)
            self.projector = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, cfg.gretriever_soft_tokens * llm_dim),
            ).to(self.device)
            if cfg.gretriever_gnn_checkpoint:
                self._load_checkpoint(cfg.gretriever_gnn_checkpoint)
                self.weight_source = f"checkpoint:{cfg.gretriever_gnn_checkpoint}"
            else:
                self.weight_source = "random_untrained_seeded"
                warnings.warn(
                    "Using an untrained GAT/MLP. This is a reproducible architecture "
                    "ablation only and must not be reported as trained G-Retriever.",
                    stacklevel=2)
            self.gnn.eval(); self.projector.eval()
        else:
            self.weight_source = "none (no GNN: Llama-2 text embedding only)"
        self.max_input_tokens = 4096

    def _make_gat(self, in_channels: int, out_channels: int, heads: int) -> nn.Module:
        if self.graph_backend == "torch_geometric_gatv2":
            return self.GATv2Conv(in_channels, out_channels, heads=heads,
                                  concat=True, add_self_loops=True)
        return self.GATv2Conv(in_channels, out_channels, heads=heads)

    def _load_checkpoint(self, path: Path) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch < 2.0
            checkpoint = torch.load(path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("The G-Retriever checkpoint must be a dictionary with 'gnn' and 'projector' weights.")
        gnn_state = checkpoint.get("gnn") or checkpoint.get("gnn_state_dict")
        projector_state = checkpoint.get("projector") or checkpoint.get("projector_state_dict")
        if not isinstance(gnn_state, dict) or not isinstance(projector_state, dict):
            raise ValueError(
                "Checkpoint must contain 'gnn'/'projector' (or '*_state_dict') keys. "
                "Do not use a bare Llama checkpoint or untrained GAT weights.")
        self.gnn.load_state_dict(gnn_state, strict=True)
        self.projector.load_state_dict(projector_state, strict=True)

    def _graph_inputs(self, audit: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_ids = [int(i) for i in audit.get("node_ids", [])]
        edge_ids = [int(i) for i in audit.get("edge_ids", [])]
        if not node_ids:
            raise ValueError("G-Retriever PCST returned no nodes.")
        original_to_local = {node_id: local for local, node_id in enumerate(node_ids)}
        x = torch.tensor(self.subgraph.node_vectors[node_ids], dtype=torch.float32, device=self.device)
        local_edges = []
        for edge_id in edge_ids:
            source, target = map(int, self.subgraph.edge_pairs[edge_id])
            if source in original_to_local and target in original_to_local:
                local_edges.append((original_to_local[source], original_to_local[target]))
        edge_index = (torch.tensor(local_edges, dtype=torch.long, device=self.device).T
                      if local_edges else torch.empty((2, 0), dtype=torch.long, device=self.device))
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        return x, edge_index, batch

    @torch.no_grad()
    def paragraph(self, keywords: list[str], audit: dict[str, Any], max_new_tokens: int) -> str:
        triples = audit.get("triples", [])[:self.cfg.gretriever_max_triples]
        if self.use_text:
            prompt = (
                "<s>[INST] You are the graph-to-text component of a medical retrieval system. "
                "Write one concise factual paragraph using ONLY the supplied graph triples. "
                "Do not answer a medical question, do not select an option, and do not add facts.\n\n"
                "Canonical ontology/KG keywords:\n- " + "\n- ".join(keywords) +
                "\n\nSelected PCST graph triples:\n- " + "\n- ".join(triples) + " [/INST]"
            )
        else:
            prompt = (
                "<s>[INST] You are the graph-to-text component of a medical retrieval system. "
                "The selected knowledge-graph subgraph is provided only as graph embeddings "
                "placed before this instruction. Write one concise factual paragraph describing "
                "the medical facts encoded by that subgraph. Do not answer a medical question, "
                "do not select an option, and do not add facts. [/INST]"
            )
        encoded = self.tok(prompt, return_tensors="pt", truncation=True,
                           max_length=self.max_input_tokens).to(self.device)
        text_embeddings = self.llm.get_input_embeddings()(encoded["input_ids"])
        if self.use_gnn:
            x, edge_index, batch = self._graph_inputs(audit)
            graph_x = torch.relu(self.gnn["gat1"](x, edge_index))
            graph_x = torch.relu(self.gnn["gat2"](graph_x, edge_index))
            graph_vector = self.global_mean_pool(graph_x, batch)
            graph_prefix = self.projector(graph_vector).view(
                1, self.cfg.gretriever_soft_tokens, self.llm.config.hidden_size).to(self.llm.dtype)
            inputs_embeds = torch.cat([graph_prefix, text_embeddings], dim=1)
            attention_mask = torch.cat([
                torch.ones((1, self.cfg.gretriever_soft_tokens), dtype=encoded["attention_mask"].dtype, device=self.device),
                encoded["attention_mask"],
            ], dim=1)
        else:
            inputs_embeds = text_embeddings
            attention_mask = encoded["attention_mask"]
        generated = self.llm.generate(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
        # Strip a textual prompt prefix only when it is demonstrably present.
        sequence = generated[0]
        textual_prompt_ids = encoded["input_ids"][0]
        if (sequence.numel() >= textual_prompt_ids.numel() and
                torch.equal(sequence[:textual_prompt_ids.numel()], textual_prompt_ids)):
            completion_ids = sequence[textual_prompt_ids.numel():]
        else:
            completion_ids = sequence
        return self.tok.decode(completion_ids, skip_special_tokens=True).strip()


class MedicalOGRAG:
    """Medical adaptation of OG-RAG's ontology-grounded hypergraph retrieval.

    A MedQuAD row becomes a factual block (one hyperedge) with ontology-grounded
    hypernodes: <condition, clinical-attribute> -> <evidence value>.  TSV KG
    triples become additional factual blocks.  At query time we retrieve top-k
    nodes by both key and value similarity, then greedily select the fewest
    factual blocks covering those nodes (Algorithm 2 in OG-RAG).
    """
    _ATTRIBUTES = {
        "symptom": ("symptom", "sign", "manifestation"),
        "cause": ("cause", "etiology", "risk factor", "why"),
        "diagnosis": ("diagnos", "test", "examination", "imaging", "mri", "ct", "eeg"),
        "treatment": ("treat", "therapy", "management", "drug", "medication"),
        "pathology": ("pathology", "histology", "biopsy", "neurofibrillary", "amyloid"),
        "prognosis": ("prognosis", "outcome", "complication", "progress"),
        "genetics": ("gene", "genetic", "inherit", "apoe"),
        "epidemiology": ("frequency", "prevalence", "population", "age"),
    }

    def __init__(self, documents: list[dict[str, str]], kg: KnowledgeGraph | None, embedder: SentenceTransformer):
        self.embedder = embedder
        self.nodes: list[tuple[str, str]] = []
        self.node_to_id: dict[tuple[str, str], int] = {}
        self.edges: list[dict[str, Any]] = []
        self.ontology_edges: list[dict[str, Any]] = []
        for doc in documents:
            text = doc["text"]
            focus = self._focus(text) or "medical condition"
            attr = self._attribute(text)
            node_ids = {self._node(f"{focus} :: has_topic", focus),
                        self._node(f"{focus} :: {attr}", self._value(text))}
            edge = {"nodes": node_ids, "text": text, "source": f"MedQuAD:{doc.get('id', '')}"}
            self.edges.append(edge); self.ontology_edges.append(edge)
        if kg:
            seen = set()
            for head, links in kg.edges.items():
                for relation, tail in links:
                    if relation.startswith("inverse:"): continue
                    signature = (head, relation, tail)
                    if signature in seen: continue
                    seen.add(signature)
                    node_ids = {self._node(f"{head} :: {relation}", tail),
                                self._node(f"{head} :: has_topic", head)}
                    self.edges.append({"nodes": node_ids,
                                       "text": f"{head} --[{relation}]--> {tail}", "source": "KG"})
        keys = [n[0] for n in self.nodes] or ["empty"]
        values = [n[1] for n in self.nodes] or ["empty"]
        self.key_embeddings = embedder.encode(keys, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        self.value_embeddings = embedder.encode(values, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        ontology_texts = [e["text"] for e in self.ontology_edges] or ["empty"]
        self.ontology_embeddings = embedder.encode(ontology_texts, normalize_embeddings=True,
                                                    convert_to_numpy=True, show_progress_bar=False).astype("float32")

    @staticmethod
    def _focus(text: str) -> str:
        m = re.search(r"Topic:\s*([^\n]+)", text)
        return m.group(1).strip() if m else ""

    @classmethod
    def _attribute(cls, text: str) -> str:
        lower = text.lower()
        for name, markers in cls._ATTRIBUTES.items():
            if any(marker in lower for marker in markers): return name
        return "clinical_information"

    @staticmethod
    def _value(text: str, limit: int = 900) -> str:
        m = re.search(r"Reference answer:\s*(.*)", text, flags=re.S)
        return (m.group(1) if m else text)[:limit].strip()

    def _node(self, key: str, value: str) -> int:
        pair = (normalise(key), normalise(value))
        if pair not in self.node_to_id:
            self.node_to_id[pair] = len(self.nodes); self.nodes.append(pair)
        return self.node_to_id[pair]

    def retrieve(self, question: str, top_k_nodes: int = 8, max_edges: int = 4,
                 online_blocks: list[dict[str, str]] | None = None,
                 force_kg: bool = False) -> tuple[str, list[dict[str, Any]]]:
        if not self.edges or not self.nodes: return "", []
        q = self.embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        online_blocks = online_blocks or []
        extra_keys = [b["key"] for b in online_blocks]
        extra_values = [b["value"] for b in online_blocks]
        if extra_keys:
            ek = self.embedder.encode(extra_keys, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
            ev = self.embedder.encode(extra_values, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
            key_embeddings = np.vstack([self.key_embeddings, ek]); value_embeddings = np.vstack([self.value_embeddings, ev])
        else:
            key_embeddings, value_embeddings = self.key_embeddings, self.value_embeddings
        offset = len(self.nodes)
        edges = list(self.edges) + [{"nodes": {offset + i}, "text": b["text"], "source": b["source"]}
                                    for i, b in enumerate(online_blocks)]
        key_scores, value_scores = key_embeddings @ q, value_embeddings @ q
        node_scores = np.maximum(key_scores, value_scores)
        k = min(top_k_nodes, len(key_embeddings))
        relevant = set(np.argsort(key_scores)[-k:]) | set(np.argsort(value_scores)[-k:])
        uncovered, selected = set(relevant), []
        while uncovered and len(selected) < max_edges:
            candidates = []
            for i, edge in enumerate(edges):
                if i in selected: continue
                covered = edge["nodes"] & uncovered
                coverage = len(covered)
                semantic = float(np.mean(node_scores[list(covered)])) if covered else float("-inf")
                candidates.append((coverage, semantic, -i, i))
            coverage, _, _, best = max(candidates, default=(0, float("-inf"), 0, -1))
            if coverage <= 0: break
            selected.append(best); uncovered -= edges[best]["nodes"]
        if force_kg and not any(edges[i]["source"] == "KG" for i in selected):
            q_norm = normalise(question)
            candidates = []
            for i, edge in enumerate(edges):
                if edge["source"] != "KG" or i in selected: continue
                subject = normalise(edge["text"].split("--", 1)[0])
                if subject and subject in q_norm:
                    candidates.append(i)
            if candidates and len(selected) < max_edges:
                candidate_texts = [edges[i]["text"] for i in candidates]
                sims = self.embedder.encode(candidate_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False) @ q
                selected.append(candidates[int(np.argmax(sims))])
        chosen = [edges[i] for i in selected]
        context = "\n\n".join(f"[OG-RAG {e['source']}]\n{e['text']}" for e in chosen)
        audit = [{"source": e["source"], "text": e["text"],
                  "hypernode_count": len(e["nodes"])} for e in chosen]
        return context, audit

    def supplemental_ontology_anchor(self, question: str, reranker: CrossEncoder,
                                     max_chars: int) -> tuple[str, dict[str, Any] | None]:
        """Return one short MedQuAD-derived ontology fact, never a KG triple."""
        if not self.ontology_edges: return "", None
        query = self.embedder.encode([question], normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)[0]
        pool = np.argsort(self.ontology_embeddings @ query)[-min(24, len(self.ontology_edges)):][::-1]
        texts = [self.ontology_edges[int(i)]["text"] for i in pool]
        scores = reranker.predict([(question, text) for text in texts], show_progress_bar=False)
        best_position = int(np.argmax(scores))
        edge, score = self.ontology_edges[int(pool[best_position])], float(scores[best_position])
        compact = edge["text"][:max_chars].strip()
        return ("[OG-RAG ontology supplement; use only if it directly supports the question]\n" + compact,
                {"source": edge["source"], "reranker_score": score, "text": compact})


class OnlineOLSOntologies:
    """Cached, opt-in access to EBI OLS (MONDO, HPO, DOID, EFO, NCIT, ChEBI).

    OLS term labels, synonyms and definitions are ontology facts, not generated
    medical advice. Failed/offline requests return an empty list and cached terms
    are reused, so a run remains reproducible after the first download.
    """
    URL = "https://www.ebi.ac.uk/ols4/api/search"
    ONTOLOGIES = ("doid", "mondo", "hp", "ncit", "chebi", "dron")
    def __init__(self, cache_path: Path, timeout: int = 45):
        self.cache_path, self.timeout, self.available = cache_path, timeout, True
        try: self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception: self.cache = {}

    @staticmethod
    def _mentions(question: str) -> list[str]:
        """Conservative exact-match candidates, not a whole noisy vignette."""
        raw = re.sub(r"\s+", " ", question).strip()
        candidates = []
        candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9'-]+(?:\s+[A-Za-z0-9'-]+){0,3}\b", raw))
        candidates.extend(re.findall(r"\b[a-z][a-z-]+(?:\s+[a-z][a-z-]+){1,3}\b", raw.lower()))
        selected, seen = [], set()
        for item in sorted(candidates, key=lambda x: (-len(x), x)):
            clean = item.strip(" ,.;:()[]")
            norm = normalise(clean)
            if len(norm) < 4 or norm in seen: continue
            if any(w in {"which", "following", "patient", "patients", "most", "with", "from", "that", "this"} for w in tokens(norm)): continue
            seen.add(norm); selected.append(clean)
            if len(selected) == 4: break
        return selected

    def _details(self, session: requests.Session, iri: str, ontology: str) -> tuple[str, list[str]]:
        try:
            # OLS4 term IRIs must be double URL encoded.
            term_iri = quote(quote(iri, safe=""), safe="")
            url = f"https://www.ebi.ac.uk/ols4/api/ontologies/{ontology}/terms/{term_iri}"
            response = session.get(url, timeout=self.timeout, headers={"Accept": "application/json"})
            if response.status_code != 200: return "", []
            data = response.json()
            desc = data.get("description") or data.get("definition") or []
            if isinstance(desc, str): desc = [desc]
            synonyms = data.get("synonyms") or data.get("synonym") or []
            if isinstance(synonyms, str): synonyms = [synonyms]
            return " ".join(str(x) for x in desc if x)[:1200], [str(x) for x in synonyms[:5]]
        except requests.RequestException:
            return "", []

    def _direct_neighbors(self, session: requests.Session, iri: str, ontology: str,
                          maximum: int) -> tuple[list[str], list[str]]:
        """Fetch only immediate OLS hierarchy neighbours, never all descendants."""
        def fetch(endpoint: str) -> list[str]:
            try:
                term_iri = quote(quote(iri, safe=""), safe="")
                url = (f"https://www.ebi.ac.uk/ols4/api/ontologies/{ontology}/terms/"
                       f"{term_iri}/{endpoint}")
                response = session.get(url, timeout=self.timeout,
                                       headers={"Accept": "application/json"})
                if response.status_code != 200: return []
                data = response.json(); embedded = data.get("_embedded", {})
                terms = (embedded.get("terms") or embedded.get(endpoint) or
                         data.get("terms") or [])
                labels, seen = [], set()
                for term in terms:
                    label = str(term.get("label") or "").strip()
                    if label and normalise(label) not in seen:
                        seen.add(normalise(label)); labels.append(label)
                    if len(labels) >= maximum: break
                return labels
            except requests.RequestException:
                return []
        with ThreadPoolExecutor(max_workers=2) as pool:
            parent = pool.submit(fetch, "hierarchicalParents")
            child = pool.submit(fetch, "hierarchicalChildren")
            return parent.result(), child.result()

    def _search_exact(self, session: requests.Session, mention: str, ontology: str) -> list[dict[str, str]]:
        try:
            response = session.get(self.URL, params={"q": mention, "ontology": ontology,
                                   "rows": 3, "exact": "true"}, timeout=self.timeout,
                                   headers={"Accept": "application/json", "User-Agent": "medical-ograg/1.0"})
            response.raise_for_status()
            docs = response.json().get("response", {}).get("docs", [])
        except requests.RequestException:
            return []
        result = []
        for doc in docs:
            label = str(doc.get("label") or "").strip()
            synonyms = doc.get("synonym") or doc.get("synonyms") or []
            if isinstance(synonyms, str): synonyms = [synonyms]
            query = normalise(mention)
            if normalise(label) != query and query not in {normalise(x) for x in synonyms}: continue
            desc = doc.get("description") or doc.get("definition") or ""
            if isinstance(desc, list): desc = " ".join(map(str, desc))
            if not desc and doc.get("iri"):
                desc, details_synonyms = self._details(session, str(doc["iri"]), ontology)
                synonyms = synonyms or details_synonyms
            value = str(desc).strip() or ("Synonyms: " + ", ".join(map(str, synonyms[:4])) if synonyms else label)
            result.append({"key": f"{ontology} :: {label} :: definition", "value": value[:1200],
                           "text": f"{label} — OLS {ontology} ontology: {value[:1200]}",
                           "source": f"OLS:{ontology}", "_iri": str(doc.get("iri") or ""),
                           "_ontology": ontology, "parents": [], "children": []})
        return result

    def lookup(self, question: str, seed_mentions: list[str] | None = None,
               fetch_neighbors: bool = False, neighbor_max: int = 2) -> list[dict[str, str]]:
        seed_mentions = seed_mentions or []
        key = (normalise(question) + "||" + "|".join(sorted(normalise(x) for x in seed_mentions)) +
               f"||ols4-neighbors-v2={int(fetch_neighbors)}:{neighbor_max}")
        if key in self.cache: return self.cache[key]
        if not self.available: return []
        try:
            mentions, seen = [], set()
            for candidate in list(seed_mentions) + self._mentions(question):
                candidate = str(candidate).strip()
                norm = normalise(candidate)
                if len(norm) < 5 or norm in seen or norm in {"protein", "cell", "gene", "patient", "patients", "model", "study"}:
                    continue
                seen.add(norm); mentions.append(candidate)
                if len(mentions) == 5: break
            if not mentions: blocks = []
            else:
                with requests.Session() as session, ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(self._search_exact, session, mention, ontology)
                               for mention in mentions for ontology in self.ONTOLOGIES]
                    blocks = [block for future in as_completed(futures) for block in future.result()]
            unique, seen = [], set()
            for block in blocks:
                fingerprint = normalise(block["key"])
                if fingerprint not in seen:
                    seen.add(fingerprint); unique.append(block)
            blocks = unique[:6]
            if fetch_neighbors and blocks:
                with requests.Session() as session, ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {pool.submit(self._direct_neighbors, session, b["_iri"], b["_ontology"], neighbor_max): b
                               for b in blocks if b.get("_iri")}
                    for future in as_completed(futures):
                        parent, child = future.result(); block = futures[future]
                        block["parents"], block["children"] = parent, child
        except Exception as exc:
            self.available = False
            print(f"OLS ontology lookup unavailable; disabled for this run ({type(exc).__name__}).")
            blocks = []
        self.cache[key] = blocks
        self.cache_path.parent.mkdir(parents=True, exist_ok=True); atomic_json(self.cache_path, self.cache)
        return blocks


def retrieval_query(question: str) -> str:
    # Options are deliberately excluded: including them makes retrieval seek distractor text.
    return question.strip()


def question_keywords_for_kg(question: str, limit: int = 24) -> list[str]:
    """Return compact question terms for KG entity matching only.

    These terms are never supplied to PCST or the graph-to-text LLM. They are
    used solely to identify semantically corresponding AD-KG entities.
    """
    stopwords = {"which", "what", "when", "where", "would", "could", "should",
                 "patient", "patients", "following", "most", "least", "with",
                 "from", "that", "this", "these", "there", "about", "after",
                 "before", "during", "because", "have", "has", "been", "were"}
    selected, seen = [], set()
    for term in tokens(question):
        if len(term) < 4 or term in stopwords or term in seen:
            continue
        seen.add(term); selected.append(term)
        if len(selected) >= limit:
            break
    return selected


def select_evidence(question: str, options: list[str], cfg: Config, kg: KnowledgeGraph | None,
                    medquad: HybridIndex | None, reranker: CrossEncoder,
                    direct_kg_facts: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    query = retrieval_query(question)
    kg_facts = (direct_kg_facts or []) if cfg.use_kg else []
    if cfg.use_kg and not direct_kg_facts and kg:
        kg_facts = kg.evidence(question, options, cfg.kg_top_k)
    hits = medquad.search(query, reranker, cfg.retrieval_pool, cfg.retrieval_top_k, cfg.min_reranker_score) if cfg.use_medquad and medquad else []
    pieces = (["Knowledge graph facts (may be irrelevant; use only if they directly fit):\n" + "\n".join(kg_facts)] if kg_facts else [])
    pieces += [f"Reference {h['id']} ({h['source']}), relevance={h['reranker_score']:.2f}:\n{h['text']}" for h in hits]
    evidence = "\n\n".join(pieces)[:cfg.max_evidence_chars]
    return evidence, {"kg_facts": kg_facts, "medquad_hits": hits, "evidence_chars": len(evidence)}


def passes_supplement_gate(question: str, direct_kg: str, candidate: str,
                           reranker: CrossEncoder, cfg: Config,
                           tolerance: float | None = None) -> tuple[bool, dict[str, float | bool]]:
    """Keep a supplement only when it is competitive with direct KG evidence.

    The gate is deliberately independent of gold labels.  With an available
    direct KG block, a candidate passes when its cross-encoder score is no more
    than ``supplement_score_tolerance`` below the direct-KG score.  The
    tolerance is a development-set hyperparameter, never a test-set knob.
    """
    if not candidate.strip():
        return False, {"enabled": bool(cfg.gate_supplements), "candidate": float("-inf")}
    if not cfg.gate_supplements or not direct_kg.strip():
        return True, {"enabled": bool(cfg.gate_supplements), "candidate": float("nan")}
    effective_tolerance = cfg.supplement_score_tolerance if tolerance is None else tolerance
    candidate_score, kg_score = map(float, reranker.predict(
        [(question, candidate), (question, direct_kg)], show_progress_bar=False))
    keep = candidate_score + effective_tolerance >= kg_score
    return keep, {"enabled": True, "candidate": candidate_score,
                  "direct_kg": kg_score, "tolerance": effective_tolerance,
                  "kept": keep}


def prompt_for_mcq(question: str, options: list[str], evidence: str, extractive: bool,
                   fewshot_examples: str = "") -> list[dict[str, str]]:
    task = ("Select the option best supported by the supplied passage. " if extractive else
            "Answer the medical multiple-choice question using clinical reasoning. ")
    system = (task + "Retrieved material can be incomplete or unrelated. Never infer an answer from a source merely because it was retrieved. "
              "Compare every option to the question; do not rely on option position. Return only the option letter.")
    user = fewshot_examples
    if user: user += "\n\n"
    user += "Question:\n" + question + "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {x}" for i, x in enumerate(options))
    if evidence: user += "\n\nRetrieved evidence (untrusted):\n" + evidence
    user += "\n\nThe best option is:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# NEW: permutation-debiased answer scoring
# --------------------------------------------------------------------------
def permutation_logprobs(model: LocalModel, question: str, options: list[str], evidence: str,
                         extractive: bool, fewshot_examples: str, k: int) -> np.ndarray:
    """Mean log-probability of every ORIGINAL option over k cyclic shifts of the option order.

    For shift s, position j of the displayed list holds original option (j+s) % n.
    The letter log-probabilities (renormalised over legal letters) are mapped back to the
    original options and averaged.  Every option therefore appears in several positions,
    which cancels the model's letter/position bias and reduces variance.  Gold labels and
    retrieval are not involved; only the display order of the same options changes.
    """
    n = len(options)
    shifts = list(range(max(1, min(k, n))))
    total = np.zeros(n, dtype="float64")
    for s in shifts:
        order = [(j + s) % n for j in range(n)]
        shown = [options[i] for i in order]
        prefix = model.chat_prefix(prompt_for_mcq(question, shown, evidence, extractive, fewshot_examples))
        lp = model.letter_logprobs(prefix, n)
        for j, original in enumerate(order):
            total[original] += lp[j]
    return total / len(shifts)


def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def predict(model: LocalModel, question: str, options: list[str], evidence: str, extractive: bool,
            mode: str, cot_max_new_tokens: int, fewshot_examples: str = "",
            perm_k: int = 4, perm_evidence_weight: float = 0.5) -> tuple[str, dict[str, float]]:
    # NEW modes: perm / perm_fusion (permutation ensemble, see module docstring).
    if mode in ("perm", "perm_fusion"):
        with_evidence = permutation_logprobs(model, question, options, evidence, extractive,
                                             fewshot_examples, perm_k)
        if mode == "perm_fusion" and evidence:
            without_evidence = permutation_logprobs(model, question, options, "", extractive,
                                                    fewshot_examples, perm_k)
            w = float(perm_evidence_weight)
            final = (1.0 - w) * _log_softmax_np(without_evidence) + w * _log_softmax_np(with_evidence)
        else:
            final = _log_softmax_np(with_evidence)
        scores = {LETTERS[i]: float(final[i]) for i in range(len(options))}
        return max(scores, key=scores.get), scores
    messages = prompt_for_mcq(question, options, evidence, extractive, fewshot_examples)
    if mode == "cot_letter":
        rationale = model.generate([
            {"role": "system", "content": "Reason medically and compare every option. Identify decisive findings. Do not guess from option position; end without giving a letter."},
            {"role": "user", "content": messages[-1]["content"]},
        ], cot_max_new_tokens)
        messages[-1]["content"] += "\n\nPrivate clinical reasoning:\n" + rationale + "\n\nNow return only the best option letter."
    if mode == "binary":
        scores = {}
        for i, candidate in enumerate(options):
            binary = [
                {"role": "system", "content": "Assess whether the candidate is the single best answer. Use the clinical facts, not its position. Answer only Yes or No."},
                {"role": "user", "content": (fewshot_examples + "\n\n" if fewshot_examples else "") + "Question:\n" + question + "\n\nCandidate answer:\n" + candidate + ("\n\nEvidence:\n" + evidence if evidence else "")},
            ]
            raw = model.score_completions(model.chat_prefix(binary), [" Yes", " No"])
            scores[LETTERS[i]] = raw[" Yes"] - raw[" No"]
        return max(scores, key=scores.get), scores
    if mode != "letter" and mode != "cot_letter":
        raise ValueError(f"unknown answer mode {mode!r}")
    prefix = model.chat_prefix(messages)
    completions = [" " + LETTERS[i] for i in range(len(options))]
    raw = model.score_completions(prefix, completions)
    scores = {c.strip(): raw[c] for c in completions}
    return max(scores, key=scores.get), scores


def item_id(item: dict[str, Any]) -> str:
    """Stable split/audit ID, including datasets that lack an explicit ID."""
    for field in ("id", "question_id", "document_id"):
        value = item.get(field)
        if value is not None and str(value).strip():
            return f"{field}:{value}"
    public = {key: value for key, value in item.items()
              if key not in {"answer", "cop", "correct_answer_id"}}
    text = json.dumps(public, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)
    return "content:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def in_split(item: dict[str, Any], split: str, dev_fraction: float) -> bool:
    if split == "all":
        return True
    bucket = int(hashlib.sha256(item_id(item).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (bucket < dev_fraction) == (split == "dev")


class FewShotSelector:
    """Select labelled demonstrations from a pool disjoint from evaluation data.

    Similarity uses question text only. Gold labels are used only after selection
    to display the known answer in a demonstration, never to rank candidates.
    """
    def __init__(self, pool: list[dict[str, Any]], processor: Processor,
                 embedder: SentenceTransformer, cfg: Config):
        self.pool, self.processor, self.embedder, self.cfg = pool, processor, embedder, cfg
        self.ids = [item_id(x) for x in pool]
        self.questions = [processor.question_choices(x)[0] for x in pool]
        self.embeddings = (embedder.encode(self.questions, normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False)
                           .astype("float32") if pool else np.empty((0, 0), dtype="float32"))

    def select(self, question: str) -> list[dict[str, Any]]:
        k = min(self.cfg.fewshot_k, len(self.pool))
        if k <= 0: return []
        if self.cfg.fewshot_strategy == "random":
            seed_text = f"{self.cfg.seed}|{normalise(question)}"
            rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
            indices = list(range(len(self.pool))); rng.shuffle(indices)
            return [self.pool[i] for i in indices[:k]]
        q = self.embedder.encode([question], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)[0]
        scores = self.embeddings @ q
        ranked = sorted(range(len(self.pool)), key=lambda i: (-float(scores[i]), self.ids[i]))
        return [self.pool[i] for i in ranked[:k]]


def format_fewshot_examples(examples: list[dict[str, Any]], processor: Processor,
                             max_chars: int) -> str:
    """Format labelled demonstrations while preserving the benchmark option order."""
    if not examples: return ""
    blocks = ["Worked examples. Their answer letters are known; use clinical reasoning rather than answer-letter frequency."]
    for i, example in enumerate(examples, 1):
        question, choices = processor.question_choices(example)
        answer = processor.gold(example)
        block = (f"Example {i}:\nQuestion: {question}\nOptions:\n" +
                 "\n".join(f"{LETTERS[j]}. {choice}" for j, choice in enumerate(choices)) +
                 f"\nCorrect answer: {answer}")
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) > max_chars:
            break
        blocks.append(block)
    return "\n\n".join(blocks) if len(blocks) > 1 else ""


def evaluate(cfg: Config) -> dict[str, Any]:
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    processor = PROCESSORS[cfg.dataset](cfg)
    data = [x for x in processor.data if in_split(x, cfg.split, cfg.dev_fraction)]
    if cfg.limit is not None: data = data[:cfg.limit]
    embedder = SentenceTransformer(cfg.embed_model, device="cpu")
    fewshot_selector = None
    pool_description = None
    if cfg.fewshot_k:
        if cfg.split == "all":
            raise ValueError("Few-shot evaluation requires --split dev or --split test so demonstrations are disjoint.")
        if cfg.fewshot_pool_file:
            raw_pool = json.loads(cfg.fewshot_pool_file.read_text(encoding="utf-8"))
            if not isinstance(raw_pool, list):
                raise ValueError("--fewshot-pool-file must contain a JSON list in the selected dataset format.")
            pool = raw_pool
            pool_description = f"file:{cfg.fewshot_pool_file}"
        else:
            if cfg.fewshot_pool_split == cfg.split:
                raise ValueError("--fewshot-pool-split must differ from --split to prevent evaluation-label leakage.")
            pool = [x for x in processor.data if in_split(x, cfg.fewshot_pool_split, cfg.dev_fraction)]
            pool_description = f"split:{cfg.fewshot_pool_split}"
        evaluated_ids, pool_ids = {item_id(x) for x in data}, {item_id(x) for x in pool}
        overlap = evaluated_ids & pool_ids
        if overlap:
            raise ValueError(f"Few-shot pool overlaps the evaluation set ({len(overlap)} IDs); refusing to run.")
        if not pool:
            raise ValueError("Few-shot pool is empty; provide --fewshot-pool-file or change --fewshot-pool-split.")
        fewshot_selector = FewShotSelector(pool, processor, embedder, cfg)
        print(f"Few-shot enabled: k={cfg.fewshot_k}; pool={pool_description}; n_pool={len(pool)}; strategy={cfg.fewshot_strategy}.")
    reranker = CrossEncoder(cfg.reranker_model, device="cpu")
    medquad_path, kg_path = cfg.data_dir / "MedQuAD.csv", cfg.data_dir / "alz.txt"
    medquad_docs = load_medquad(medquad_path) if medquad_path.exists() else []
    medquad = HybridIndex(medquad_docs, embedder) if cfg.use_medquad else None
    kg = KnowledgeGraph(kg_path) if (cfg.use_kg or cfg.use_ograg or cfg.use_gretriever_subgraph or cfg.use_gretriever or cfg.use_gretriever_gnn) and kg_path.exists() else None
    sparql_context = SparqlKGContext(kg, embedder) if kg and cfg.kg_context_mode == "sparql" else None
    ograg = MedicalOGRAG(medquad_docs, kg, embedder) if cfg.use_ograg else None
    gretriever_subgraph = GRetrieverSubgraph(kg, embedder) if (cfg.use_gretriever_subgraph or cfg.use_gretriever or cfg.use_gretriever_gnn) and kg else None
    gretriever_llm = None
    gretriever_gnn_llm = None
    if cfg.use_gretriever or cfg.use_gretriever_gnn:
        if not cfg.use_ograg:
            raise ValueError("G-Retriever is an OG-RAG extension; also pass --use-ograg.")
        if cfg.use_gretriever and cfg.use_gretriever_gnn:
            raise ValueError("Choose one graph-to-text mode: --use-gretriever or --use-gretriever-gnn.")
        if cfg.use_gretriever:
            gretriever_llm = KeywordOnlyGRetrieverLLM(cfg)
        else:
            gretriever_gnn_llm = TrainedGNNGraphToTextLLM(cfg, gretriever_subgraph)
    ols = OnlineOLSOntologies(cfg.output_dir / "ols_ontology_cache.json", cfg.ols_timeout) if cfg.online_ontologies else None
    model = LocalModel(cfg)
    rows, failures = [], []
    for n, item in enumerate(data, 1):
        try:
            question, choices = processor.question_choices(item)
            if not 2 <= len(choices) <= 5: raise ValueError(f"unsupported number of choices: {len(choices)}")
            demonstrations = fewshot_selector.select(question) if fewshot_selector else []
            fewshot_text = format_fewshot_examples(demonstrations, processor, cfg.fewshot_max_chars)
            sparql_facts, sparql_audit = (sparql_context.retrieve(question, reranker, cfg)
                                          if sparql_context else ([], {}))
            evidence, audit = select_evidence(question, choices, cfg, kg, medquad, reranker, sparql_facts)
            if sparql_context:
                audit["sparql_kg_context"] = sparql_audit
            audit["fewshot"] = {"enabled": bool(fewshot_selector), "k_requested": cfg.fewshot_k,
                                "k_used": len(demonstrations), "pool": pool_description if fewshot_selector else None,
                                "strategy": cfg.fewshot_strategy if fewshot_selector else None,
                                "demo_ids": [item_id(x) for x in demonstrations]}
            if ograg:
                kg_seeds = []
                if kg and cfg.ols_seed_kg_entities:
                    q_norm = normalise(question)
                    kg_seeds = [label for label in kg.labels
                                if len(normalise(label)) >= 5 and normalise(label) in q_norm]
                    kg_seeds = sorted(kg_seeds, key=lambda x: -len(normalise(x)))[:5]
                online_blocks = ols.lookup(question, kg_seeds, cfg.ols_fetch_neighbors,
                                           cfg.ols_neighbor_max) if ols else []
                if online_blocks:
                    ols_scores = reranker.predict(
                        [(question, str(block["text"])) for block in online_blocks],
                        show_progress_bar=False)
                    online_blocks = [block for block, _ in sorted(
                        zip(online_blocks, ols_scores),
                        key=lambda item: float(item[1]), reverse=True)]
                    audit["ols_reranker_scores"] = [float(s) for s in sorted(ols_scores, reverse=True)]
                audit["ols_seed_mentions"] = kg_seeds
                og_context, og_edges = ograg.retrieve(question, top_k_nodes=8, max_edges=4,
                                                       online_blocks=online_blocks, force_kg=cfg.ograg_force_kg)
                selected_hyperedges = [e for e in og_edges if e.get("source") != "KG"]
                direct_kg = ""
                if sparql_context:
                    direct_kg = "\n".join(sparql_facts)
                elif kg:
                    direct_kg = "\n".join(kg.evidence(question, choices, cfg.kg_top_k))
                chosen_source = "ograg"
                if cfg.ograg_safe_fallback and direct_kg:
                    og_context, chosen_source = direct_kg, "direct_kg_safe"
                elif cfg.ograg_abstain_without_kg and not direct_kg:
                    og_context, chosen_source = "", "no_weak_ontology_evidence"
                elif cfg.ograg_evidence_router and direct_kg:
                    scores = reranker.predict([(question, og_context), (question, direct_kg)], show_progress_bar=False)
                    if float(scores[0]) < float(scores[1]) + cfg.ograg_router_margin:
                        og_context, chosen_source = direct_kg, "direct_kg"
                    audit["router_scores"] = {"ograg": float(scores[0]), "direct_kg": float(scores[1])}
                if cfg.ograg_append_ontology_anchor and direct_kg:
                    if online_blocks:
                        ols_anchor = online_blocks[0]["text"][:cfg.ontology_anchor_chars].strip()
                        parents = online_blocks[0].get("parents", [])[:cfg.ols_neighbor_max]
                        children = online_blocks[0].get("children", [])[:cfg.ols_neighbor_max]
                        if parents: ols_anchor += "\nDirect ontology parents: " + "; ".join(parents)
                        if children: ols_anchor += "\nDirect ontology children: " + "; ".join(children)
                        keep, gate = passes_supplement_gate(
                            question, direct_kg, ols_anchor, reranker, cfg,
                            tolerance=cfg.ols_supplement_tolerance)
                        audit["ols_supplement_gate"] = gate
                        if keep:
                            og_context = (og_context + "\n\n[OLS ontology supplement; exact term match]\n" + ols_anchor).strip()
                            audit["ols_supplement"] = {"source": online_blocks[0]["source"], "text": ols_anchor,
                                                        "parents": parents, "children": children}
                    if cfg.ograg_append_medquad_anchor:
                        anchor, anchor_audit = ograg.supplemental_ontology_anchor(
                            question, reranker, cfg.ontology_anchor_chars)
                        if anchor:
                            keep, gate = passes_supplement_gate(question, direct_kg, anchor, reranker, cfg)
                            audit["medquad_supplement_gate"] = gate
                            if keep:
                                og_context = (og_context + "\n\n" + anchor).strip()
                                audit["medquad_supplement"] = anchor_audit
                if cfg.ograg_append_hyperedges and direct_kg and selected_hyperedges:
                    text = "\n\n".join(f"[{e['source']}]\n{e['text']}" for e in selected_hyperedges)
                    text = text[:cfg.ograg_hyperedge_chars].strip()
                    if text:
                        keep, gate = passes_supplement_gate(question, direct_kg, text, reranker, cfg)
                        audit["ograg_hyperedge_gate"] = gate
                        if keep:
                            og_context = (og_context + "\n\n[OG-RAG selected hyperedges]\n" + text).strip()
                            audit["appended_hyperedges"] = selected_hyperedges
                use_graph_paragraph = bool(gretriever_subgraph) and (
                    bool(direct_kg) or not cfg.gretriever_require_direct_kg)
                if use_graph_paragraph:
                    question_terms = question_keywords_for_kg(question)
                    entity_matches = gretriever_subgraph.match_question_keywords_to_entities(
                        question_terms, cfg.gretriever_entity_seeds,
                        cfg.gretriever_entity_min_similarity)
                    graph_keywords = [str(match["entity"]) for match in entity_matches]
                    subgraph_text, graph_audit = gretriever_subgraph.retrieve(
                        graph_keywords,
                        cfg.gretriever_topk_nodes if (cfg.use_gretriever or cfg.use_gretriever_gnn) else cfg.gretriever_max_nodes,
                        cfg.gretriever_max_edges, cfg.gretriever_chars)
                    if cfg.gretriever_rerank_triples and graph_audit.get("triples"):
                        triple_scores = reranker.predict(
                            [(question, triple) for triple in graph_audit["triples"]],
                            show_progress_bar=False)
                        ranked_triples = sorted(
                            zip(graph_audit["triples"], triple_scores),
                            key=lambda item: float(item[1]), reverse=True)
                        ranked_triples = ranked_triples[:cfg.gretriever_rerank_top_k]
                        graph_audit["triples"] = [triple for triple, _ in ranked_triples]
                        graph_audit["post_pcst_reranker_scores"] = [float(score) for _, score in ranked_triples]
                        graph_audit["post_pcst_reranking"] = True
                        subgraph_text = ("[G-Retriever selected subgraph]\n" +
                                         "\n".join(graph_audit["triples"]))[:cfg.gretriever_chars]
                    if cfg.use_gretriever_subgraph and subgraph_text:
                        og_context = (og_context + "\n\n" + subgraph_text).strip()
                    audit["gretriever_subgraph"] = graph_audit
                    audit["gretriever_pcst_keyword_source"] = "cosine_matched_ad_kg_entities_only"
                    audit["gretriever_question_keywords"] = question_terms
                    audit["gretriever_entity_matches"] = entity_matches
                    if gretriever_llm and graph_audit.get("triples"):
                        generated = gretriever_llm.paragraph(
                            graph_audit["keywords"], graph_audit["triples"],
                            cfg.gretriever_max_new_tokens)[:cfg.gretriever_paragraph_chars].strip()
                        keep, gate = passes_supplement_gate(
                            question, direct_kg, generated, reranker, cfg,
                            tolerance=cfg.gretriever_supplement_tolerance)
                        if generated and keep:
                            og_context = (og_context + "\n\n[G-Retriever Llama-2 graph paragraph]\n" + generated).strip()
                        audit["gretriever_llama2"] = {"used": bool(generated and keep), "gate": gate,
                            "keywords": graph_audit["keywords"], "triples": graph_audit["triples"],
                            "paragraph": generated}
                    if gretriever_gnn_llm and graph_audit.get("triples"):
                        generated = gretriever_gnn_llm.paragraph(
                            graph_audit["keywords"], graph_audit,
                            cfg.gretriever_max_new_tokens)[:cfg.gretriever_paragraph_chars].strip()
                        keep, gate = passes_supplement_gate(
                            question, direct_kg, generated, reranker, cfg,
                            tolerance=cfg.gretriever_supplement_tolerance)
                        if generated and keep:
                            og_context = (og_context + "\n\n[G-Retriever GNN--Llama-2 graph paragraph]\n" + generated).strip()
                        audit["gretriever_gnn"] = {"used": bool(generated and keep), "gate": gate,
                            "embedding_mode": gretriever_gnn_llm.mode,
                            "keywords": graph_audit["keywords"], "triples": graph_audit["triples"],
                            "paragraph": generated,
                            "weight_source": gretriever_gnn_llm.weight_source,
                            "graph_backend": gretriever_gnn_llm.graph_backend}
                elif gretriever_subgraph:
                    audit["gretriever_subgraph"] = {"used_pcst": False,
                        "skipped_reason": "no_direct_question_grounded_kg_fact"}
                evidence = (og_context + ("\n\n" + evidence if evidence else ""))[:cfg.max_evidence_chars]
                audit["ograg_edges"] = og_edges
                audit["ols_blocks_found"] = len(online_blocks)
                audit["selected_evidence_source"] = chosen_source
            pred, scores = predict(model, question, choices, evidence, processor.IS_EXTRACTIVE,
                                   cfg.answer_mode, cfg.cot_max_new_tokens, fewshot_text,
                                   cfg.perm_k, cfg.perm_evidence_weight)
            gold = processor.gold(item)
            rows.append({"id": item.get("id", item.get("question_id", n)), "prediction": pred, "gold": gold,
                         "correct": pred == gold, "score_by_letter": scores, "question": question,
                         "choices": choices, "retrieval": audit})
        except Exception as exc:
            failures.append({"index": n, "id": item.get("id", ""), "error": f"{type(exc).__name__}: {exc}"})
            rows.append({"id": item.get("id", n), "prediction": "", "correct": False, "error": failures[-1]["error"]})
        if n % cfg.save_every == 0: atomic_json(cfg.output_dir / f"{cfg.dataset}_{cfg.split}_predictions.json", rows)
        if n % 25 == 0: print(f"{n}/{len(data)}; accuracy={sum(bool(r.get('correct')) for r in rows)/len(rows):.3f}")
    accuracy = sum(bool(r.get("correct")) for r in rows) / len(rows) if rows else 0.0
    by_retrieval = {}
    for name, predicate in {"kg_used": lambda r: bool(r.get("retrieval", {}).get("kg_facts")) or any(e.get("source") == "KG" for e in r.get("retrieval", {}).get("ograg_edges", [])),
                            "sparql_context_used": lambda r: bool(r.get("retrieval", {}).get("sparql_kg_context", {}).get("selected_triples")),
                            "medquad_used": lambda r: bool(r.get("retrieval", {}).get("medquad_hits")),
                            "ograg_used": lambda r: bool(r.get("retrieval", {}).get("ograg_edges")),
                            "ols_used": lambda r: bool(r.get("retrieval", {}).get("ols_blocks_found", 0)),
                            "ograg_kg_used": lambda r: any(e.get("source") == "KG" for e in r.get("retrieval", {}).get("ograg_edges", [])),
                            "router_direct_kg": lambda r: str(r.get("retrieval", {}).get("selected_evidence_source", "")).startswith("direct_kg"),
                            "medquad_supplement_used": lambda r: bool(r.get("retrieval", {}).get("medquad_supplement")),
                            "ols_supplement_used": lambda r: bool(r.get("retrieval", {}).get("ols_supplement")),
                            "ograg_hyperedges_appended": lambda r: bool(r.get("retrieval", {}).get("appended_hyperedges")),
                            "ols_neighbors_used": lambda r: bool(r.get("retrieval", {}).get("ols_supplement", {}).get("parents") or r.get("retrieval", {}).get("ols_supplement", {}).get("children")),
                            "gretriever_subgraph_used": lambda r: bool(r.get("retrieval", {}).get("gretriever_subgraph", {}).get("triples")),
                            "gretriever_llama2_used": lambda r: bool(r.get("retrieval", {}).get("gretriever_llama2", {}).get("used")),
                            "gretriever_gnn_used": lambda r: bool(r.get("retrieval", {}).get("gretriever_gnn", {}).get("used")),
                            "fewshot_used": lambda r: bool(r.get("retrieval", {}).get("fewshot", {}).get("k_used", 0))}.items():
        group = [r for r in rows if predicate(r)]
        by_retrieval[name] = {"n": len(group), "accuracy": sum(bool(r.get("correct")) for r in group)/len(group) if group else None}
    summary = {"config": asdict(cfg, dict_factory=lambda x: {k: str(v) if isinstance(v, Path) else v for k, v in x}),
               "n": len(rows), "accuracy": accuracy, "failures": failures, "diagnostics": by_retrieval,
               "warning": "This is a measured result for this exact split/configuration, not a guaranteed target."}
    atomic_json(cfg.output_dir / f"{cfg.dataset}_{cfg.split}_summary.json", summary)
    return summary


def parse_args() -> Config:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default="."); p.add_argument("--output-dir", default="results")
    p.add_argument("--dataset", choices=PROCESSORS, default="medqa"); p.add_argument("--model-name", default=Config.model_name)
    p.add_argument("--main-device", default=Config.main_device, help="auto, cpu, cuda:0, or cuda:1 for final Llama-3.")
    p.add_argument("--embed-model", default=Config.embed_model); p.add_argument("--reranker-model", default=Config.reranker_model)
    p.add_argument("--limit", type=int); p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--dev-fraction", type=float, default=.15); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrieval-top-k", type=int, default=3); p.add_argument("--retrieval-pool", type=int, default=30)
    p.add_argument("--kg-top-k", type=int, default=6); p.add_argument("--max-evidence-chars", type=int, default=5000)
    p.add_argument("--kg-context-mode", choices=["direct", "sparql"], default="sparql",
                   help="Direct KG context: legacy lexical triples or cosine-seeded SPARQL paths and local neighbourhoods.")
    p.add_argument("--sparql-entity-seeds", type=int, default=2,
                   help="Number of cosine-matched AD-KG entity seeds; two are used for bounded path construction.")
    p.add_argument("--sparql-min-similarity", type=float, default=0.35,
                   help="Minimum question-to-AD-KG cosine similarity for a SPARQL seed.")
    p.add_argument("--sparql-max-hops", type=int, choices=[1, 2, 3], default=3,
                   help="Maximum number of SPARQL path hops between the two seeds.")
    p.add_argument("--sparql-max-paths", type=int, default=4,
                   help="Maximum bounded paths retained before neighbour exploration.")
    p.add_argument("--sparql-neighbor-limit", type=int, default=12,
                   help="Maximum immediate SPARQL neighbours queried per expanded entity.")
    p.add_argument("--sparql-neighbor-expand-k", type=int, default=3,
                   help="Number of query-relevant first-hop neighbours expanded by one additional hop.")
    p.add_argument("--sparql-rerank-top-k", type=int, default=6,
                   help="Maximum SPARQL path/neighbour triples retained after cross-encoder plus MMR reranking.")
    p.add_argument("--sparql-mmr-lambda", type=float, default=0.75,
                   help="MMR relevance/diversity weight in [0,1] used after the cross-encoder reranker.")
    p.add_argument("--max-input-tokens", type=int, default=8192); p.add_argument("--min-reranker-score", type=float, default=0.0)
    p.add_argument("--no-kg", action="store_true"); p.add_argument("--no-medquad", action="store_true")
    p.add_argument("--use-kg", action="store_true", help="Opt in to KG evidence; first verify it helps on dev.")
    p.add_argument("--use-medquad", action="store_true", help="Opt in to MedQuAD evidence; first verify it helps on dev.")
    p.add_argument("--use-ograg", action="store_true", help="Use OG-RAG-style ontology-grounded hypergraph retrieval.")
    p.add_argument("--online-ontologies", action="store_true", help="Download and cache matching ontology facts from EBI OLS.")
    p.add_argument("--no-ols-seed-kg-entities", action="store_true", help="Do not use question-grounded KG labels as exact OLS search mentions.")
    p.add_argument("--ols-fetch-neighbors", action="store_true", help="Fetch direct OLS parents and children for exact ontology matches.")
    p.add_argument("--ols-neighbor-max", type=int, default=2, help="Maximum direct parents and children retained per OLS concept.")
    p.add_argument("--use-gretriever-subgraph", action="store_true", help="Append a G-Retriever-style PCST subgraph paragraph built directly from alz.txt.")
    p.add_argument("--gretriever-max-nodes", type=int, default=12, help="Maximum nodes in the selected G-Retriever subgraph.")
    p.add_argument("--gretriever-max-edges", type=int, default=12, help="Maximum triples rendered in its paragraph.")
    p.add_argument("--gretriever-chars", type=int, default=1000, help="Maximum characters of the rendered G-Retriever subgraph paragraph.")
    p.add_argument("--gretriever-rerank-triples", action="store_true",
                   help="Post-PCST rerank selected triples against the question before graph-to-text generation; answer choices are excluded.")
    p.add_argument("--gretriever-rerank-top-k", type=int, default=6,
                   help="Maximum PCST triples retained after post-PCST cross-encoder reranking.")
    p.add_argument("--use-gretriever", action="store_true", help="Use Llama-2 as a second, keyword-only PCST graph-to-text stage.")
    p.add_argument("--use-gretriever-gnn", action="store_true",
                   help="Use a trained GAT plus trained MLP soft-prefix with Llama-2 for PCST graph textualization.")
    p.add_argument("--gretriever-embedding-mode", choices=["both", "gnn", "llm"], default="llm",
                   help="Subgraph embedding(s) passed to Llama-2: 'both' = GNN + LLM text embedding (original), "
                        "'gnn' = GNN soft-prefix embedding only, 'llm' = Llama-2 text embedding only (no GNN/checkpoint needed).")
    p.add_argument("--gretriever-gnn-checkpoint",
                   help="Optional checkpoint containing trained 'gnn' and 'projector' state dictionaries.")
    p.add_argument("--gretriever-untrained-gnn", action="store_true",
                   help="Ablation only: initialize the GAT/MLP from the seeded random state instead of loading a checkpoint.")
    p.add_argument("--gretriever-gnn-hidden-dim", type=int, default=384,
                   help="Hidden dimension used when the GAT/projector checkpoint was trained.")
    p.add_argument("--gretriever-gnn-heads", type=int, default=4,
                   help="Number of attention heads used when the GAT checkpoint was trained.")
    p.add_argument("--gretriever-soft-tokens", type=int, default=8,
                   help="Number of trained graph-prefix embeddings supplied to Llama-2.")
    p.add_argument("--gretriever-model-name", default=Config.gretriever_model_name)
    p.add_argument("--gretriever-device", default=Config.gretriever_device, help="cpu (safe default), cuda:0, or cuda:1.")
    p.add_argument("--gretriever-topk-nodes", type=int, default=50)
    p.add_argument("--gretriever-topk-edges", type=int, default=3)
    p.add_argument("--gretriever-entity-seeds", type=int, default=8,
                   help="Maximum AD-KG entities selected by cosine similarity before PCST.")
    p.add_argument("--gretriever-entity-min-similarity", type=float, default=0.35,
                   help="Minimum cosine similarity for a question-keyword-to-KG-entity match.")
    p.add_argument("--gretriever-max-new-tokens", type=int, default=96)
    p.add_argument("--gretriever-max-triples", type=int, default=30)
    p.add_argument("--gretriever-paragraph-chars", type=int, default=700,
                   help="Hard maximum size of the generated graph paragraph.")
    p.add_argument("--gretriever-allow-without-direct-kg", action="store_true",
                   help="Ablation: allow graph generation even without a question-grounded KG fact.")
    p.add_argument("--ols-timeout", type=int, default=45, help="Seconds for one OLS request; two attempts are made.")
    p.add_argument("--ograg-force-kg", action="store_true", help="Include one question-grounded KG factual block in OG-RAG if greedy coverage omitted it.")
    p.add_argument("--ograg-evidence-router", action="store_true", help="Choose between OG-RAG context and directly-grounded KG evidence using the medical reranker.")
    p.add_argument("--ograg-router-margin", type=float, default=0.35, help="Extra reranker score OG-RAG needs to beat direct KG evidence.")
    p.add_argument("--ograg-safe-fallback", action="store_true", help="Keep direct KG evidence whenever available; use OG-RAG only for KG coverage gaps.")
    p.add_argument("--ograg-abstain-without-kg", action="store_true", help="Do not inject generic OG-RAG text when no KG fact is grounded in the question.")
    p.add_argument("--ograg-append-ontology-anchor", action="store_true", help="Append one short reranked exact OLS block after direct KG evidence.")
    p.add_argument("--ograg-append-medquad-anchor", action="store_true",
                   help="Separately append one short MedQuAD reference block; this is disabled by default so OLS results are not mixed with MedQuAD text.")
    p.add_argument("--ontology-anchor-chars", type=int, default=360, help="Maximum characters of the supplementary ontology block.")
    p.add_argument("--ograg-append-hyperedges", action="store_true", help="Append selected non-KG OG-RAG hyperedges after the direct KG context.")
    p.add_argument("--ograg-hyperedge-chars", type=int, default=700, help="Maximum characters of appended OG-RAG hyperedges.")
    p.add_argument("--gate-supplements", action="store_true",
                   help="Use the cross-encoder to exclude OLS, OG-RAG, and G-Retriever supplements that are weak relative to direct KG evidence.")
    p.add_argument("--supplement-score-tolerance", type=float, default=0.25,
                   help="Development-tuned score gap allowed below direct KG evidence for a supplement to remain in context.")
    p.add_argument("--ols-supplement-tolerance", type=float,
                   help="Optional OLS-only gate tolerance. A negative value requires OLS to outrank direct KG evidence; omit to use --supplement-score-tolerance.")
    p.add_argument("--gretriever-supplement-tolerance", type=float,
                   help="Optional G-Retriever-only gate tolerance; omit to use --supplement-score-tolerance.")
    p.add_argument("--answer-mode", choices=["letter", "cot_letter", "binary", "perm", "perm_fusion"], default="letter",
                   help="perm = permutation-debiased letter scoring; perm_fusion = fuse evidence-free and "
                        "evidence-conditioned permutation ensembles (recommended).")
    p.add_argument("--cot-max-new-tokens", type=int, default=180)
    p.add_argument("--perm-k", type=int, default=4,
                   help="Number of cyclic option shifts for perm / perm_fusion (capped at the number of options).")
    p.add_argument("--perm-evidence-weight", type=float, default=0.5,
                   help="w in final = (1-w)*no_evidence + w*with_evidence (perm_fusion). Tune on dev only.")
    p.add_argument("--fewshot-k", type=int, default=0,
                   help="Number of labelled, disjoint demonstrations to prepend; 0 disables few-shot prompting.")
    p.add_argument("--fewshot-pool-split", choices=["dev", "test"], default="dev",
                   help="Disjoint hash split used as the demonstration pool when --fewshot-pool-file is omitted.")
    p.add_argument("--fewshot-pool-file",
                   help="Optional JSON demonstration pool in the selected dataset's original format. It must not overlap evaluation IDs.")
    p.add_argument("--fewshot-strategy", choices=["semantic", "random"], default="semantic",
                   help="Select demonstrations by question similarity or deterministic random sampling.")
    p.add_argument("--fewshot-max-chars", type=int, default=1800,
                   help="Maximum total characters allocated to few-shot demonstrations.")
    p.add_argument("--enable-web", action="store_true", help="Reserved opt-in; web evidence is disabled in this reproducible baseline.")
    p.add_argument("--ablations", action="store_true", help="Run baseline, no-KG, no-MedQuAD, and no-retrieval variants in separate folders.")
    a = p.parse_args()
    return Config(data_dir=Path(a.data_dir), output_dir=Path(a.output_dir), model_name=a.model_name, main_device=a.main_device,
                  embed_model=a.embed_model, reranker_model=a.reranker_model, dataset=a.dataset, limit=a.limit,
                  split=a.split, dev_fraction=a.dev_fraction, seed=a.seed, retrieval_top_k=a.retrieval_top_k,
                  retrieval_pool=a.retrieval_pool, kg_top_k=a.kg_top_k, max_evidence_chars=a.max_evidence_chars,
                  kg_context_mode=a.kg_context_mode,
                  sparql_entity_seeds=a.sparql_entity_seeds,
                  sparql_min_similarity=a.sparql_min_similarity,
                  sparql_max_hops=a.sparql_max_hops,
                  sparql_max_paths=a.sparql_max_paths,
                  sparql_neighbor_limit=a.sparql_neighbor_limit,
                  sparql_neighbor_expand_k=a.sparql_neighbor_expand_k,
                  sparql_rerank_top_k=a.sparql_rerank_top_k,
                  sparql_mmr_lambda=a.sparql_mmr_lambda,
                  max_input_tokens=a.max_input_tokens, min_reranker_score=a.min_reranker_score,
                  use_kg=a.use_kg and not a.no_kg, use_medquad=a.use_medquad and not a.no_medquad,
                  use_ograg=a.use_ograg, enable_web=a.enable_web, ablations=a.ablations, answer_mode=a.answer_mode,
                  cot_max_new_tokens=a.cot_max_new_tokens, online_ontologies=a.online_ontologies,
                  perm_k=a.perm_k, perm_evidence_weight=a.perm_evidence_weight,
                  ols_timeout=a.ols_timeout, ograg_force_kg=a.ograg_force_kg,
                  ograg_evidence_router=a.ograg_evidence_router, ograg_router_margin=a.ograg_router_margin,
                  ograg_safe_fallback=a.ograg_safe_fallback,
                  ograg_abstain_without_kg=a.ograg_abstain_without_kg,
                  ograg_append_ontology_anchor=a.ograg_append_ontology_anchor,
                  ograg_append_medquad_anchor=a.ograg_append_medquad_anchor,
                  ontology_anchor_chars=a.ontology_anchor_chars,
                  ols_seed_kg_entities=not a.no_ols_seed_kg_entities,
                  ograg_append_hyperedges=a.ograg_append_hyperedges,
                  ograg_hyperedge_chars=a.ograg_hyperedge_chars,
                  gate_supplements=a.gate_supplements,
                  supplement_score_tolerance=a.supplement_score_tolerance,
                  ols_supplement_tolerance=a.ols_supplement_tolerance,
                  gretriever_supplement_tolerance=a.gretriever_supplement_tolerance,
                  ols_fetch_neighbors=a.ols_fetch_neighbors,
                  ols_neighbor_max=a.ols_neighbor_max,
                  use_gretriever_subgraph=a.use_gretriever_subgraph,
                  gretriever_max_nodes=a.gretriever_max_nodes,
                  gretriever_max_edges=a.gretriever_max_edges,
                  gretriever_chars=a.gretriever_chars,
                  gretriever_rerank_triples=a.gretriever_rerank_triples,
                  gretriever_rerank_top_k=a.gretriever_rerank_top_k,
                  use_gretriever=a.use_gretriever,
                  use_gretriever_gnn=a.use_gretriever_gnn,
                  gretriever_embedding_mode=a.gretriever_embedding_mode,
                  gretriever_untrained_gnn=a.gretriever_untrained_gnn,
                  gretriever_gnn_checkpoint=Path(a.gretriever_gnn_checkpoint) if a.gretriever_gnn_checkpoint else None,
                  gretriever_gnn_hidden_dim=a.gretriever_gnn_hidden_dim,
                  gretriever_gnn_heads=a.gretriever_gnn_heads,
                  gretriever_soft_tokens=a.gretriever_soft_tokens,
                  gretriever_model_name=a.gretriever_model_name,
                  gretriever_device=a.gretriever_device,
                  gretriever_topk_nodes=a.gretriever_topk_nodes,
                  gretriever_topk_edges=a.gretriever_topk_edges,
                  gretriever_entity_seeds=a.gretriever_entity_seeds,
                  gretriever_entity_min_similarity=a.gretriever_entity_min_similarity,
                  gretriever_max_new_tokens=a.gretriever_max_new_tokens,
                  gretriever_max_triples=a.gretriever_max_triples,
                  gretriever_paragraph_chars=a.gretriever_paragraph_chars,
                  gretriever_require_direct_kg=not a.gretriever_allow_without_direct_kg,
                  fewshot_k=a.fewshot_k, fewshot_pool_split=a.fewshot_pool_split,
                  fewshot_pool_file=Path(a.fewshot_pool_file) if a.fewshot_pool_file else None,
                  fewshot_strategy=a.fewshot_strategy, fewshot_max_chars=a.fewshot_max_chars)


if __name__ == "__main__":
    config = parse_args()
    if config.enable_web:
        print("Web retrieval remains disabled by design in this baseline: it is non-reproducible and can introduce source/version leakage.")
    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN is not set. This is fine for a public/cached model; a gated model requires it via the environment or Kaggle Secrets.")
    if not config.ablations:
        result = evaluate(config)
        print(json.dumps({k: result[k] for k in ("n", "accuracy", "diagnostics", "warning")}, indent=2))
    else:
        # Run this on --split dev while choosing thresholds/model variants.  Do not
        # select the best row after looking at an untouched test split.
        variants = {"baseline": (config.use_kg, config.use_medquad),
                    "no_kg": (False, config.use_medquad),
                    "no_medquad": (config.use_kg, False),
                    "no_retrieval": (False, False)}
        report = {}
        parent = config.output_dir
        for name, (use_kg, use_medquad) in variants.items():
            variant = Config(**{**asdict(config), "output_dir": parent / name,
                                "use_kg": use_kg, "use_medquad": use_medquad, "ablations": False})
            result = evaluate(variant)
            report[name] = {"accuracy": result["accuracy"], "n": result["n"], "diagnostics": result["diagnostics"]}
        atomic_json(parent / f"{config.dataset}_{config.split}_ablation_summary.json", report)
        print(json.dumps(report, indent=2))
        