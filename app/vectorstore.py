"""Vector store — load from disk (populated by ingest.py).

Production pattern:
  ingest.py  → loads docs, splits, embeds, saves to persist_dir   (run once / on doc change)
  this file  → loads existing embeddings from disk, no re-embedding (every app startup)

Startup is instant because embeddings are already on disk.
"""

import os
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from app.config import settings

_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings() -> FastEmbedEmbeddings:
    """Return the shared embedding function (ONNX-based, no PyTorch)."""
    return FastEmbedEmbeddings(model_name=_EMBEDDING_MODEL)


def load_vectorstore() -> Chroma:
    """Load the persisted Chroma DB from disk.

    Raises FileNotFoundError if ingest.py has not been run yet.
    """
    if not os.path.exists(settings.persist_dir):
        raise FileNotFoundError(
            f"Vector store not found at '{settings.persist_dir}'. "
            "Run `python ingest.py` first to ingest your documents."
        )
    return Chroma(
        persist_directory=settings.persist_dir,
        embedding_function=get_embeddings(),
        collection_name="production_rag",
    )


def get_retriever(vectorstore: Chroma):
    """Return a retriever that fetches top-k similar chunks."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_k},
    )
