"""Entry point.

IMPORTANT: Run `python ingest.py` before starting the server.
The server will fail with FileNotFoundError if chroma_db does not exist.

Start: uvicorn main:app --reload
Docs:  http://localhost:8000/docs
"""

from fastapi import FastAPI
from app.api import router
from app.config import settings

app = FastAPI(
    title="Production RAG Pipeline",
    description=(
        "Production-grade RAG agent with document ingestion, relevance filtering, "
        "hallucination verification, and quality-gated retries."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "model": settings.groq_model, "persist_dir": settings.persist_dir}
