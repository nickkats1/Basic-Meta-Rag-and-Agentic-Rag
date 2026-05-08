# SEC RAG Pipeline

A multi-strategy Retrieval-Augmented Generation toolkit for querying SEC
filings (or any PDF). Ships with four interchangeable retrieval strategies, a
metrics suite for ranking and answer quality, an evaluation harness, and a
multi-provider LLM factory that supports HuggingFace models directly via the
`huggingface_hub` / `transformers` libraries (no LangChain wrapper required).

## Features

### Retrieval strategies (`rag.retrievers`)

| Strategy | Module | When to use |
| --- | --- | --- |
| `BM25Retriever` | `bm25_retriever` | Pure keyword / sparse retrieval. Fast, no embeddings, strong for exact terms. |
| `DenseRetriever` | `dense_retriever` | Bi-encoder + FAISS. Captures paraphrased / semantic matches. |
| `HybridRetriever` | `hybrid_retriever` | BM25 + dense, fused with Reciprocal Rank Fusion. Best of both. |
| `RerankerRetriever` | `reranker` | Wraps any base retriever with a cross-encoder rerank pass. Highest precision. |

All four implement the same `BaseRetriever` interface (`add_documents`,
`retrieve`), so they're swap-compatible inside `RAGPipeline`.

### Metrics (`rag.metrics`)

* Retrieval: `hit_rate_at_k`, `precision_at_k`, `recall_at_k`,
  `mean_reciprocal_rank`, `average_precision`, `mean_average_precision`,
  `ndcg_at_k`.
* Generation (embedding-based, no LLM judge): `answer_similarity`,
  `embedding_faithfulness`, `context_precision`.

### Evaluation (`rag.evaluation`)

`evaluate_retriever(...)` and `evaluate_pipeline(...)` run a labeled dataset
through any retriever / pipeline and return aggregate + per-example metrics.

### LLM providers (`rag.llm`)

| Provider key | Backend | Library |
| --- | --- | --- |
| `openai` | OpenAI Chat Completions | `langchain-openai` |
| `groq` | Groq | `langchain-groq` |
| `google` | Gemini | `langchain-google-genai` |
| `huggingface` | HF Inference API | `huggingface_hub` (no LangChain) |
| `huggingface_local` | Local `transformers.pipeline` | `transformers` (no LangChain) |

`RAGPipeline` accepts either flavor.

## Setup

```bash
git clone https://github.com/nickkats1/Sec-Rag.git
cd Sec-Rag
python -m venv venv
source venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

Installing the package also exposes a `sec-rag` console script equivalent to
`python -m rag.cli`.

Add a `.env` file in the project root with whichever provider keys you plan
to use (the file is gitignored):

```
GROQ_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
HF_TOKEN=...
```

Drop one or more PDFs into `data/`. The `data/` directory is gitignored, so
the filings stay local even though SEC 10-K filings are public documents and
could legally be committed -- keeping them out of git just keeps the repo
small:

```
data/
├── 10-K_CompanyA.pdf
├── 10-Q_CompanyB.pdf
```

## Run from the command line

The CLI has two subcommands: `answer` (single query, one retriever) and
`compare` (same query through all four retrievers, with retrieval metrics).

```bash
python -m rag.cli answer \
    --pdf data/google_10K.pdf \
    --retriever reranker \
    --provider groq \
    --model llama-3.3-70b-versatile \
    --query "What was total revenue in each fiscal year reported?"
```

```bash
# BM25-only baseline
python -m rag.cli answer --pdf data/google_10K.pdf --retriever bm25 \
    --query "Operating margin for every reported year"

# Hybrid + Gemini
python -m rag.cli answer --pdf data/google_10K.pdf --retriever hybrid \
    --provider google --model gemini-2.0-flash \
    --query "Cash and marketable securities at year end"

# Local HuggingFace model
python -m rag.cli answer --pdf data/google_10K.pdf --retriever dense \
    --provider huggingface_local \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --query "Number of full-time employees at year end"
```

Add `--show-contexts` to print the retrieved chunks before the answer.

```bash
# Score all four retrievers on the same query
python -m rag.cli compare --pdf data/google_10K.pdf \
    --query "Operating income for each reported year" --top-k 5
```

Without `--gold-id`, `compare` uses the cross-encoder reranker as a stand-in
oracle (which biases metrics toward the reranker). Pass `--gold-id chunk_42`
(repeatable) for hand-labeled relevance.

## Run the notebooks

The `notebooks/` directory has one notebook per retrieval strategy plus an
evaluation notebook. They expect a PDF at `data/google_10K.pdf`; change the
`FILE_PATH` constant in the first cell to point at any other filing.

```bash
jupyter lab notebooks/
# or, headless re-execution of all five:
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

| Notebook | Topic |
| --- | --- |
| `01_bm25.ipynb` | Pure BM25 sparse retrieval |
| `02_dense.ipynb` | Bi-encoder + FAISS, with PCA/KMeans embedding plot |
| `03_hybrid.ipynb` | BM25 + dense with Reciprocal Rank Fusion |
| `04_reranker.ipynb` | Cross-encoder reranking on top of hybrid |
| `05_evaluation.ipynb` | Side-by-side retrieval metrics for all four |

## Programmatic use

```python
import os
from rag.data_ingestion import load_documents, chunk_documents
from rag.retrievers import HybridRetriever, RerankerRetriever
from rag.llm import LLM
from rag.pipeline import RAGPipeline

docs = load_documents("data/10-K_CompanyA.pdf")
chunks = chunk_documents(docs, chunk_size=2000, chunk_overlap=200)
# `chunk_documents` auto-assigns `metadata['doc_id']` ("chunk_0", "chunk_1", ...).

retriever = RerankerRetriever(HybridRetriever(), candidate_pool=20)
retriever.add_documents(chunks)

llm = LLM(api_key=os.environ["GROQ_API_KEY"]).get_llm(
    provider="groq",
    model_name="llama-3.3-70b-versatile",
    temperature=0.0,
)
pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=5)
print(pipeline.answer("What was total revenue in fiscal 2024?").answer)
```

Tilt `HybridRetriever` toward sparse or dense matches with
`HybridRetriever(bm25_weight=1.0, dense_weight=2.0)`. Persist a built dense
index with `DenseRetriever.save("path/")` and reload with
`DenseRetriever.load("path/")` to skip re-embedding on subsequent runs.

### HuggingFace, the direct way

```python
from rag.huggingface_llm import HuggingFaceAPILLM, HuggingFaceLocalLLM

# Cloud (HF serverless API):
llm = HuggingFaceAPILLM(model_name="mistralai/Mistral-7B-Instruct-v0.2",
                       api_token=os.environ["HF_TOKEN"])

# Local (transformers pipeline):
llm = HuggingFaceLocalLLM(model_name="HuggingFaceTB/SmolLM2-360M-Instruct")
```

## Evaluation

```python
from rag.evaluation import EvalExample, evaluate_pipeline

dataset = [
    EvalExample(
        question="What was total revenue in 2024?",
        relevant_doc_ids=["chunk_42"],
        reference_answer="Revenue was $400 billion.",
    ),
]
report = evaluate_pipeline(pipeline, dataset, k=5)
print(report.metrics)
# {'hit@5': 1.0, 'precision@5': 0.2, 'recall@5': 1.0, 'ndcg@5': 1.0,
#  'mrr': 1.0, 'answer_similarity': 0.91, 'faithfulness': 0.87,
#  'context_precision': 0.4, 'map': 1.0}
```

## Tests

```bash
pytest -q
```

The retrieval and pipeline tests mock the bi-encoder / cross-encoder loaders
so they run offline.

## A note on SEC filings in the repo

10-K, 10-Q, and other SEC EDGAR filings are public documents -- there's no
licensing or confidentiality issue with committing them. The `data/`
directory is gitignored anyway because the PDFs are large (often 5-10MB
each) and would bloat clones. Treat the gitignore as a "keep the repo small"
rule, not a "this is sensitive" rule.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.
