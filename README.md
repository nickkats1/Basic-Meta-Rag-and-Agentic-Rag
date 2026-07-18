# SEC RAG Pipeline

[![CI](https://github.com/nickkats1/Sec-Rag/actions/workflows/ci.yaml/badge.svg)](https://github.com/nickkats1/Sec-Rag/actions/workflows/ci.yaml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ask any SEC 10-K or 10-Q filing a question in plain English and get an answer grounded in the actual text—with the retrieved passages included so you can verify the LLM's work.

This isn't just one RAG pipeline; it's **four retrieval strategies side by side**, sharing a common interface for honest comparison:

- **BM25** — pure keyword search (fast, no ML)
- **Dense** — semantic embeddings (captures paraphrases)
- **Hybrid** — combines both with Reciprocal Rank Fusion
- **Reranker** — cross-encoder re-ranking for highest precision

All four speak the same language, so switching from one to another is a one-line change. An evaluation harness benchmarks all four against hand-labeled ground truth and reports the raw numbers—no cherry-picking.

## Real Results: Does Reranking Actually Help?

Five retrieval strategies benchmarked on Alphabet's FY2025 10-K (`data/google_10K.pdf`, 433 chunks at default `chunk_size=1000`), scored on a small query set with keyword-based relevance labels.

| Retriever | NDCG@5 | MRR | Precision@5 | Recall@5 | MAP |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0.710 | 1.000 | 0.600 | 0.279 | 0.909 |
| Dense | 0.676 | 0.875 | 0.600 | 0.224 | 0.863 |
| Hybrid (RRF) | 0.754 | 0.875 | 0.700 | 0.313 | 0.819 |
| Dense + Rerank | 0.775 | 0.833 | 0.750 | 0.326 | 0.808 |
| **Hybrid + Rerank** | **0.780** | 0.833 | **0.750** | **0.326** | 0.821 |

**The story:** Reranking wins on ranking quality — Hybrid + Rerank posts the best NDCG@5, Precision@5, and Recall@5. But BM25 takes MRR outright: the test queries are keyword-heavy (financial line items), which is BM25's home turf. No single strategy dominates every metric — which is exactly why the harness reports all of them. See [notebooks/05_evaluation.ipynb](notebooks/05_evaluation.ipynb) for the full run.

**A note on honesty:** the ground truth labels a chunk relevant if it contains every key phrase of the query — a heuristic, not human judgment. Treat absolute numbers with skepticism; the *relative* comparison between retrievers on the same labels is the meaningful signal.

## Features

### Retrieval strategies

All share one interface (`add_documents()`, `retrieve()`), so swapping is one line:

| Strategy | When to use |
| --- | --- |
| `BM25Retriever` | Fast keyword search, no embeddings |
| `DenseRetriever` | Semantic matching with FAISS |
| `HybridRetriever` | BM25 + dense with Reciprocal Rank Fusion |
| `RerankerRetriever` | Cross-encoder re-ranking for highest precision |

### Metrics

- Retrieval: Hit@K, Precision, Recall, NDCG, MRR, MAP
- Generation: Answer similarity (embedding cosine between generated and reference answers)

Use `evaluate_retriever()` or `evaluate_pipeline()` to score on labeled datasets.

### LLM providers

| Provider | Backend | Notes |
| --- | --- | --- |
| `openai` | OpenAI | Via langchain-openai |
| `groq` | Groq | Via langchain-groq |
| `google` | Gemini | Via langchain-google-genai |
| `huggingface_local` | Local transformers | Direct via transformers, no API key needed |

Switch providers without touching retrieval code.

## How it works

**The pipeline flow:**

1. Load PDF → extract text → split into overlapping chunks (default 1000 tokens, 100 overlap)
2. Pick a retriever and add chunks to its index
3. User asks a question
4. Retriever finds top-K most relevant chunks
5. LLM reads those chunks and answers the question
6. User gets the answer + the chunks so they can verify

**Why four retrievers?**

- **BM25** is fast but can only match exact keywords (misses "revenue" if you ask about "sales")
- **Dense** captures meaning via embeddings but is slower and can hallucinate connections
- **Hybrid** runs both in parallel, combining precision and recall
- **Reranker** takes hybrid's results and re-ranks them with a fine-tuned cross-encoder for maximum accuracy (but higher latency)

Pick by use case: fast baseline (BM25) → balanced (Hybrid) → maximum accuracy (Reranker). The evaluation harness lets you measure the tradeoff on your data.

## Setup

```bash
git clone https://github.com/nickkats1/Sec-Rag.git && cd Sec-Rag
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"  # runtime + test/notebook/viz dependencies
```

Without `[dev]`, use `pip install -e .` for runtime only.

**Add API keys** (`.env` is gitignored):

```dotenv
GROQ_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
HF_TOKEN=...
```

**Add PDFs** to `data/` (directory is gitignored to keep repo small):

```text
data/
├── google_10K.pdf
├── other_10K.pdf
```

## Command-line usage

Two subcommands: `answer` (single query) and `compare` (all four retrievers on one query).

**Single answer:**

```bash
python -m cli answer \
  --pdf data/google_10K.pdf \
  --retriever hybrid \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --query "What was total revenue in 2024?"
```

**Examples:**

```bash
# BM25 baseline
python -m cli answer --pdf data/google_10K.pdf --retriever bm25 \
  --query "Operating margin for every reported year"

# Dense embeddings
python -m cli answer --pdf data/google_10K.pdf --retriever dense \
  --provider huggingface_local \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --query "Employee count at year end"

# Show retrieved chunks before the answer
python -m cli answer --pdf data/google_10K.pdf --retriever reranker \
  --provider groq --model llama-3.3-70b-versatile \
  --show-contexts --query "Your question here"
```

**Compare all four retrievers:**

```bash
python -m cli compare --pdf data/google_10K.pdf \
  --query "Operating income per year" --top-k 5
```

Without `--gold-id`, `compare` just prints each retriever's top-K doc IDs. Pass `--gold-id chunk_42` (repeatable) to score Hit/Precision/Recall against labeled ground truth.

## Notebooks

Five notebooks in `notebooks/`, one per retriever + evaluation, all executed with outputs saved. All expect `data/google_10K.pdf`; edit `PDF_PATH` in the first cell to use another filing.

```bash
cd notebooks
jupyter lab .
```

Or re-execute all:

```bash
for nb in notebooks/*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

| Notebook | Topic |
| --- | --- |
| `01_bm25.ipynb` | BM25 keyword search |
| `02_dense.ipynb` | Dense embeddings + FAISS, with index persistence |
| `03_hybrid.ipynb` | BM25 + dense fusion (Reciprocal Rank Fusion) |
| `04_reranker.ipynb` | Cross-encoder re-ranking |
| `05_evaluation.ipynb` | Side-by-side metrics (source of the table above) |

## Programmatic usage

```python
import os
from rag import (
    load_documents,
    chunk_documents,
    HybridRetriever,
    RerankerRetriever,
    RAGPipeline,
    get_llm,
)

docs = load_documents("data/google_10K.pdf")
chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=100)

retriever = RerankerRetriever(base_retriever=HybridRetriever())
retriever.add_documents(chunks)

llm = get_llm(
    provider="groq",
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
)

pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=5)
result = pipeline.answer("What was total revenue in 2024?")
print(result.answer)
print(result.contexts)  # Retrieved chunks
```

**Tune hybrid retriever:**

```python
# Bias toward dense or sparse
hybrid = HybridRetriever(bm25_weight=1.0, dense_weight=2.0)
```

**Persist dense index:**

```python
# Save after embedding (the dense half of the hybrid retriever)
hybrid.dense.save("saved_index/")

# Load to skip re-embedding next time
hybrid.dense.load("saved_index/")
```

## Evaluation

Use the same harness that produced the table above on any dataset:

```python
from rag import evaluate_pipeline, EvalExample

dataset = [
    EvalExample(
        question="What was total revenue in 2024?",
        relevant_doc_ids=["chunk_42"],
        reference_answer="Revenue was $400 billion.",  # Optional
    ),
]

report = evaluate_pipeline(pipeline, dataset, k=5)
print(report.metrics)
# {'hit@5': 1.0, 'precision@5': 0.2, 'recall@5': 1.0, 'ndcg@5': 1.0,
#  'mrr': 1.0, 'map': 1.0, 'answer_similarity': 0.91}
```

Omit `reference_answer` for retrieval metrics only (no LLM call).

## Testing

```bash
pytest -q
```

Tests use the real (small) sentence-transformers models; the first run downloads them (~90 MB), after which everything runs from the local cache. No LLM API calls are made.

## Troubleshooting

**PDFs fail to load:**

- Ensure the file exists at the path you passed
- Text extraction uses `pypdf`; it's included in `pip install -e .`

**Embeddings are slow on first run:**

- Models download on first use (usually 300MB–1GB)
- Store them in `HF_HOME` or cache directory to reuse
- Use `DenseRetriever.save()` to persist the index after build

**API key not found:**

- Check your `.env` file exists and has `PROVIDER_API_KEY=value` (no quotes, no spaces)
- If using `os.environ[]` directly, the variable must be set before running

**Reranker scores are low on small queries:**

- Cross-encoders need context; if your query + chunks are too short, scores may not discriminate
- Verify the candidate pool isn't bottlenecking: the reranker can only reorder what the base retriever surfaces (`04_reranker.ipynb` explores pool-size tradeoffs)

## About SEC filings in this repo

10-K and 10-Q filings are public documents—no licensing concerns. The `data/` directory is gitignored because PDFs are large (5–10MB) and bloat clones, not because they're sensitive.

## License

MIT License — see [LICENSE](LICENSE)
