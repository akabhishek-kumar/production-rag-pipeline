# production-rag-pipeline

A **production-grade RAG pipeline** with persistent vector storage, relevance filtering, hallucination verification, and RAGAS-style evaluation. Built with LangGraph, LangChain, Groq, and Chroma.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph 0.2+ |
| LLM | Groq — llama-3.1-8b-instant (free tier) |
| Embeddings | FastEmbed BAAI/bge-small-en-v1.5 (ONNX) |
| Vector Store | Chroma (persistent on disk) |
| Document Loaders | LangChain — .txt, .pdf, .docx |
| API | FastAPI + uvicorn |

## What Makes This "Production"

| Feature | Basic RAG | This Pipeline |
|---------|-----------|---------------|
| Vector store | In-memory | **Persistent Chroma on disk** |
| Ingestion | Inline | **Separate `ingest.py`** |
| Relevance check | None | **Pre-generation filtering** |
| Hallucination check | None | **Post-generation verification** |
| Evaluation | None | **RAGAS-style `evaluate.py`** |
| No-result handling | Hallucinate | **"No information found" response** |

## Architecture — 7-Node LangGraph Pipeline

```
Question
   │
   ▼
[retrieve] ── fetches top-k chunks from persistent Chroma
   │
   ▼
[filter_docs] ── LLM grades each chunk for relevance (JsonOutputParser)
   │
   ├── no relevant docs ──► [no_info] ──► END
   │
   ▼
[generate] ── RAG chain with filtered context only
   │
   ▼
[verify_answer] ── checks answer is grounded in retrieved docs (hallucination guard)
   │
   ▼
[evaluate] ── grades answer quality 1-10
   │
   ├── score ≥ threshold ──► END
   │
   └── score < threshold ──► [rewrite] ──► [retrieve]  (max retries)
```

## Project Structure

```
app/
├── config.py       # chunk_size, persist_dir, grade_threshold, max_retries
├── vectorstore.py  # load persistent Chroma (raises error if not ingested)
├── chains.py       # 5 chains: rag, relevance, hallucination, grade, rewrite
├── graph.py        # 7-node LangGraph with conditional routing
└── api.py          # POST /query/ endpoint
ingest.py           # load docs/ → chunk → embed → save to chroma_db/
evaluate.py         # RAGAS-style: context_relevance, faithfulness, quality, recall
docs/               # drop your .txt / .pdf / .docx files here
main.py             # FastAPI entry point
```

## Key Concepts Demonstrated

- **Separate ingestion pipeline** — `ingest.py` runs once; `main.py` loads persisted store
- **Relevance filtering** — removes irrelevant chunks before generation (reduces hallucination)
- **Hallucination verification** — independent post-generation check, not self-evaluated
- **JsonOutputParser pattern** — used for boolean fields (Groq `with_structured_output` fails on booleans)
- **Harness Engineering** — all 5 components: Tool Registry, Model Management, Context Management, Guardrails, Verification Steps
- **RAGAS-style evaluation** — measures context relevance, answer faithfulness, keyword recall

## Quick Start

```bash
git clone https://github.com/akabhishek-kumar/production-rag-pipeline
cd production-rag-pipeline
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # add your Groq API key

# Step 1 — ingest your documents
# Drop .txt/.pdf/.docx files into docs/ folder, then:
python ingest.py

# Step 2 — start the API
uvicorn main:app --reload
```

## API

```bash
POST /query/
{
  "question": "What are UiPath Coded Agents?"
}
```

## Evaluate Pipeline Quality

```bash
python evaluate.py
```

Outputs scores for context relevance, answer faithfulness, answer quality, and keyword recall across test questions.

---

Part of my AI Engineering learning series → [GitHub](https://github.com/akabhishek-kumar)
