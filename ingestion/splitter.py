"""
Text splitting — uses LangChain's RecursiveCharacterTextSplitter to break
each loaded page into overlapping chunks small enough for good embedding
quality, while preserving the source/page metadata on every chunk.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import logger

# Legal text tends to have long clauses; 800 chars (~150-200 words) with
# 150 chars overlap keeps most single clauses intact while still being
# small enough for precise retrieval.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of page-level Documents into smaller overlapping chunks.
    Metadata (source, page, file) is automatically carried over to each
    resulting chunk by LangChain's splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks
