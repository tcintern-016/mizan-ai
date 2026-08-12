"""
Pakistan Law Assistant — Data Ingestion Pipeline.

Run this once (or whenever data/raw/ changes):
    python ingestion/ingest.py
    python ingestion/ingest.py --force     # re-ingest even if data already exists

Pipeline: Load PDFs (LangChain PyPDFLoader)
       -> Split into chunks (LangChain RecursiveCharacterTextSplitter)
       -> Embed (HuggingFace sentence-transformers, local, free)
       -> Store in ChromaDB (persisted to disk)
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is on sys.path so `from utils.config import ...` works
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from utils.embeddings import LightweightEmbeddings
from langchain_chroma import Chroma

from ingestion.loader import load_all_documents
from ingestion.splitter import split_documents
from utils.config import settings
from utils.logger import logger


def get_existing_count() -> int:
    """Check how many chunks already exist in the ChromaDB collection, if any."""
    if not settings.chroma_persist_path.exists():
        return 0
    try:
        embeddings = LightweightEmbeddings()
        vectordb = Chroma(
            persist_directory=str(settings.chroma_persist_path),
            embedding_function=embeddings,
            collection_name=settings.chroma_collection_name,
        )
        return vectordb._collection.count()
    except Exception:
        return 0


def main(force: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("Pakistan Law Assistant — Starting Data Ingestion")
    logger.info("=" * 60)

    existing = get_existing_count()
    if existing > 0 and not force:
        logger.info(
            f"ChromaDB already has {existing} chunks. "
            "Pass --force to re-ingest from scratch."
        )
        return

    # 1. Load
    documents = load_all_documents()
    if not documents:
        logger.error(
            "No documents were loaded. Place PDFs in data/raw/ "
            "(see ingestion/loader.py DOCUMENT_MAP for expected filenames)."
        )
        return

    # 2. Split
    chunks = split_documents(documents)

    # 3. Embed
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    embeddings = LightweightEmbeddings()

    # 4. Store in ChromaDB
    logger.info(f"Storing {len(chunks)} chunks in ChromaDB at {settings.chroma_persist_path} …")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_persist_path),
        collection_name=settings.chroma_collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

    logger.info("=" * 60)
    logger.info(f"Ingestion complete. {len(chunks)} chunks stored.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-ingest even if data exists")
    args = parser.parse_args()
    main(force=args.force)
