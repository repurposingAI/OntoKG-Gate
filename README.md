```markdown
# OntoKG-Gate

OntoKG-Gate is an evidence-gated framework for Alzheimer’s disease medical multiple-choice question answering. It combines a disease-specific knowledge graph, semantic embeddings, online ontology enrichment, OG-RAG hypergraph retrieval, optional G-Retriever subgraph contextualization, and permutation-debiased LLM answer scoring.

The framework uses direct AD-KG evidence as the primary context. Ontology, OG-RAG, and G-Retriever evidence are added only when a relevance gate determines that they provide useful supplementary information.

---

## 1. Main Components

### 1.1 Direct AD-KG Retrieval

The framework retrieves question-grounded evidence from the Alzheimer’s disease knowledge graph using:

- question keyword extraction;
- cosine similarity between question keywords and KG entities;
- SPARQL path retrieval;
- local neighbor exploration;
- cross-encoder reranking;
- maximal marginal relevance (MMR) for compact and non-redundant triples.

The retrieved direct KG facts remain the primary evidence source.

### 1.2 Online Ontology Enrichment

The framework can query the EBI Ontology Lookup Service (OLS) to retrieve exact biomedical concept matches from ontologies such as:

- MONDO;
- DOID;
- HPO;
- EFO;
- NCIT;
- ChEBI.

Ontology blocks may include preferred labels, definitions, synonyms, and optional direct hierarchical parents or children.

### 1.3 OG-RAG Hypergraph Retrieval

OG-RAG represents KG triples and ontology blocks as factual hyperedges. It retrieves query-relevant hypernodes using key and value similarity, then applies greedy hyperedge coverage to retain a compact set of relevant evidence blocks.

### 1.4 Evidence Gating

The evidence gate compares the relevance of direct KG evidence with supplementary OLS, OG-RAG, or G-Retriever evidence using a cross-encoder.

Direct KG facts are preserved. Supplementary evidence is appended only when it satisfies the configured relevance criterion.

### 1.5 G-Retriever Subgraph Contextualization

G-Retriever extracts a compact subgraph from AD-KG using cosine-matched KG entity seeds and Prize-Collecting Steiner Tree (PCST) optimization.

The selected triples can be transformed into a factual paragraph by a graph-to-text Llama-2 model. The final Llama-3 model receives this paragraph only when the evidence gate accepts it.

### 1.6 Permutation-Debiased Answer Scoring

The framework can cyclically permute answer options before scoring them with the final LLM. Scores are mapped back to the original options and averaged.

This reduces answer-position bias, because each answer content appears in multiple letter positions.

---

## 2. Repository Structure

```text
OntoKG-Gate/
│
├── main.py
│   Main evaluation and inference script.
│
├── knowledge-graph/
│   └── alz.txt
│       Alzheimer’s disease knowledge graph in TSV format.
│       Each line follows: head <TAB> relation <TAB> tail.
│
├── kg-embedding/
│   └── entity_embeddings.pkl
│       Precomputed embeddings for AD-KG entities.
│
├── keyword-embeddings/
│   └── keyword_embeddings.pkl
│       Precomputed embeddings for biomedical keywords.
│
├── datasets/
│   ├── medqa_filter.json
│   ├── medmcqa_filter.json
│   ├── mmlu_filter.json
│   └── qa4mre_filter.json
│
└── results/
    Output directory created during evaluation.
```

---

## 3. Supported Datasets

The framework supports four Alzheimer’s disease-related medical QA datasets:

| Dataset | Input Format | Number of Options |
|---|---|---:|
| MedQA | Medical multiple-choice questions | 4 |
| MedMCQA | Medical entrance examination questions | 4 |
| MMLU-medical | Medical subsets of MMLU | 4 |
| QA4MRE-AD | Alzheimer’s disease reading comprehension questions | 5 |

The dataset labels are used only for final evaluation. They are never used during KG retrieval, ontology lookup, reranking, PCST selection, evidence gating, or prompt construction.

---

## 4. Installation

Install the required packages:

```bash
pip install torch transformers sentence-transformers rank-bm25 rdflib requests numpy
```

Optional packages:

```bash
pip install pcst-fast torch-geometric
```

## 5. Hugging Face Authentication

For gated Hugging Face models, set your token as an environment variable:

```bash
export HF_TOKEN="your_hugging_face_token"
```

Do not store Hugging Face tokens directly inside `main.py`.

---

## 6. Usage

Here is the command I used to run the approach on the QA4MRE test dataset :

```bash
python3 main.py --data-dir ./ --dataset qa4mre --split test --kg-context-mode sparql --sparql-entity-seeds 2 --sparql-min-similarity 0.45 --sparql-max-hops 3 --sparql-max-paths 4 --sparql-neighbor-limit 10 --sparql-neighbor-expand-k 3 --sparql-rerank-top-k 6 --sparql-mmr-lambda 0.75 --use-ograg --online-ontologies --ols-timeout 10 --ograg-force-kg --ograg-safe-fallback --ograg-abstain-without-kg --ograg-append-ontology-anchor --ontology-anchor-chars 120 --use-gretriever-gnn --gretriever-embedding-mode both --gretriever-gnn-checkpoint gnn_ckpt.pt --gretriever-device cpu --main-device cuda:0 --gretriever-entity-seeds 4 --gretriever-entity-min-similarity 0.45 --gretriever-topk-nodes 16 --gretriever-max-edges 4 --gretriever-chars 320 --gretriever-paragraph-chars 280 --gretriever-max-new-tokens 32 --gretriever-rerank-triples --gretriever-rerank-top-k 4 --gate-supplements --supplement-score-tolerance 0.10 --ols-supplement-tolerance -0.40 --gretriever-supplement-tolerance -0.15 --answer-mode perm_fusion --perm-k 4 --perm-evidence-weight 0.5 --output-dir results/test/qa4mre/full_gnn
```



## 7. Output Files

Each evaluation creates the following files:

```text
results/<experiment_name>/
├── <dataset>_<split>_predictions.json
├── <dataset>_<split>_summary.json
└── ols_ontology_cache.json
```

The prediction file records:

- question identifier;
- predicted answer;
- gold answer;
- correctness;
- score for each option letter;
- SPARQL seed entities;
- selected KG triples;
- OLS blocks;
- OG-RAG hyperedges;
- gate decisions;
- G-Retriever subgraph details;
- few-shot demonstration identifiers.

The summary file records accuracy, failures, configuration, and component-level diagnostics.

---
