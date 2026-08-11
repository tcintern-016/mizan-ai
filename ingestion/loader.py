"""
Document loading — uses LangChain's PyPDFLoader to load each legal PDF
page-by-page, preserving page numbers as metadata (needed for citations).

Expected files in data/raw/ (rename downloaded PDFs to match, or edit
DOCUMENT_MAP below):

    constitution.pdf     -> "Constitution of Pakistan"
    ppc.pdf               -> "Pakistan Penal Code"
    contract_act.pdf       -> "Contract Act 1872"
    peca.pdf                -> "PECA 2016 (Prevention of Electronic Crimes Act)"
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from utils.config import settings
from utils.logger import logger

# Filename (in data/raw/) -> human-readable source label used in citations
DOCUMENT_MAP = {
    "constitution.pdf": "Constitution of Pakistan",
    "ppc.pdf": "Pakistan Penal Code",
    "contract_act.pdf": "Contract Act 1872",
    "peca.pdf": "PECA 2016 (Prevention of Electronic Crimes Act)",
}


def load_all_documents() -> List[Document]:
    """
    Load every PDF listed in DOCUMENT_MAP that actually exists in data/raw/.
    Missing files are skipped with a warning (so you can start with just
    one or two documents and add more later).

    Returns a flat list of LangChain Document objects, one per PDF page,
    each with metadata: {"source": <label>, "page": <int>, "file": <filename>}.
    """
    raw_dir = settings.data_raw_path
    all_docs: List[Document] = []

    if not raw_dir.exists():
        logger.error(f"data/raw directory not found at {raw_dir}")
        return all_docs

    for filename, label in DOCUMENT_MAP.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            logger.warning(f"Skipping missing file: {filename} (expected at {filepath})")
            continue

        logger.info(f"Loading {filename} as '{label}' …")
        loader = PyPDFLoader(str(filepath))
        pages = loader.load()  # one Document per PDF page

        for page_doc in pages:
            # PyPDFLoader already sets metadata['page'] (0-indexed); normalize to 1-indexed
            page_num = page_doc.metadata.get("page", 0) + 1
            page_doc.metadata = {
                "source": label,
                "page": page_num,
                "file": filename,
            }
            all_docs.append(page_doc)

        logger.info(f"  -> {len(pages)} pages loaded from {filename}")

    logger.info(f"Total pages loaded across all documents: {len(all_docs)}")
    return all_docs


def load_single_document(filename: str, label: str) -> List[Document]:
    """Load one specific PDF by filename, useful for incremental ingestion."""
    filepath = settings.data_raw_path / filename
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist")

    loader = PyPDFLoader(str(filepath))
    pages = loader.load()
    for page_doc in pages:
        page_num = page_doc.metadata.get("page", 0) + 1
        page_doc.metadata = {"source": label, "page": page_num, "file": filename}
    return pages
