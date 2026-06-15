"""Ingestion pipeline — run this once (or whenever your documents change).

Usage:
  python ingest.py

What it does:
  1. Reads every .txt, .pdf, .docx file from DOCS_DIR (./docs by default)
  2. Splits each document into overlapping chunks
  3. Embeds all chunks using FastEmbed (ONNX, no PyTorch)
  4. Saves embeddings to PERSIST_DIR (./chroma_db by default)

The main app (main.py) just loads from chroma_db — no re-embedding on startup.
"""

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.vectorstore import get_embeddings

try:
    from langchain_community.document_loaders.word_document import Docx2txtLoader
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


def load_documents(docs_dir: str) -> list:
    """Load all supported documents from the docs folder."""
    docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"Docs folder '{docs_dir}' not found. Creating it.")
        docs_path.mkdir(parents=True)
        return docs

    for file_path in sorted(docs_path.iterdir()):
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
            print(f"  Loaded: {file_path.name}")
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())
            print(f"  Loaded: {file_path.name} ({len(loader.load())} pages)")
        elif file_path.suffix == ".docx" and DOCX_AVAILABLE:
            loader = Docx2txtLoader(str(file_path))
            docs.extend(loader.load())
            print(f"  Loaded: {file_path.name}")
        else:
            print(f"  Skipped: {file_path.name} (unsupported type)")

    return docs


def split_documents(docs: list) -> list:
    """Split documents into chunks.

    RecursiveCharacterTextSplitter splits on paragraph breaks first,
    then sentence breaks, then word breaks — preserving semantic units
    as much as possible within the chunk_size limit.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_and_persist(chunks: list) -> None:
    """Embed chunks and save to disk."""
    embeddings = get_embeddings()

    # If a DB already exists, delete it and rebuild fresh
    # (for incremental updates you'd want upsert logic instead)
    import shutil
    if os.path.exists(settings.persist_dir):
        shutil.rmtree(settings.persist_dir)
        print(f"  Cleared existing store at {settings.persist_dir}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="production_rag",
        persist_directory=settings.persist_dir,
    )
    print(f"  Saved {len(chunks)} chunks to {settings.persist_dir}")
    return vectorstore


def main():
    print(f"\nIngestion pipeline starting")
    print(f"  Docs dir  : {settings.docs_dir}")
    print(f"  Persist   : {settings.persist_dir}")
    print(f"  Chunk size: {settings.chunk_size} chars, overlap: {settings.chunk_overlap}")
    print()

    print("Step 1: Loading documents...")
    docs = load_documents(settings.docs_dir)
    if not docs:
        print("  No documents found. Add .txt, .pdf, or .docx files to ./docs and re-run.")
        return
    print(f"  Loaded {len(docs)} document(s)")

    print("\nStep 2: Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    print("\nStep 3: Embedding and saving to disk...")
    print("  (First run downloads the ONNX model ~40MB — subsequent runs are instant)")
    build_and_persist(chunks)

    print("\nIngestion complete. Run `uvicorn main:app --reload` to start the API.")


if __name__ == "__main__":
    main()
