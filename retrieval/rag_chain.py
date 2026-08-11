"""
RAG chain — retrieval + LLM generation for the Pakistan Law Assistant.

Uses Groq (cloud, fast, free-tier friendly) as the LLM via langchain-groq.
Built with LangChain Expression Language (LCEL) so both the generated
answer AND the raw retrieved chunks (for citation display) come back
from a single chain invocation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env before anything reads os.environ
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retrieval.retriever import LawRetriever, RetrievedChunk
from utils.config import settings
from utils.logger import logger

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


def _format_docs(docs: List[Document]) -> str:
    """Format retrieved LangChain Documents into a numbered context block."""
    if not docs:
        return "No relevant legal text was found for this question."
    lines = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "?")
        lines.append(f"[{i}] ({source}, p.{page})\n{doc.page_content}\n")
    return "\n".join(lines)


class LawRAGChain:
    """RAG chain combining a LangChain retriever with a Groq LLM."""

    def __init__(self) -> None:
        self.law_retriever = LawRetriever()
        self.llm = self._build_llm()
        self.prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
        self.output_parser = StrOutputParser()

    def _build_llm(self):
        groq_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "or your deployment's environment variables."
            )
        logger.info(f"Using Groq LLM ({settings.groq_model})")
        return ChatGroq(
            api_key=groq_key,
            model=settings.groq_model,
            temperature=0.1,
            max_tokens=1024,
        )

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> dict:
        """
        Run the full RAG pipeline: retrieve -> format context -> generate.

        Returns a dict with the answer text, the source citations used,
        and the raw context string sent to the LLM (useful for debugging
        and for the "show retrieved sections" bonus feature).
        """
        chunks = self.law_retriever.retrieve_with_scores(
            question, top_k=top_k, source_filter=source_filter
        )

        # Rebuild lightweight Documents from the scored chunks so the
        # same formatting function can be reused.
        docs = [
            Document(page_content=c.text, metadata={"source": c.source, "page": c.page})
            for c in chunks
        ]
        context_str = _format_docs(docs)

        logger.info(f"Querying LLM with {len(docs)} retrieved chunks")
        try:
            chain = self.prompt | self.llm | self.output_parser
            answer = chain.invoke({"context": context_str, "question": question})
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            answer = (
                "I apologize, but I was unable to generate a response right now. "
                "Please verify your GROQ_API_KEY is set correctly and try again.\n\n"
                "⚠️ This response is for educational and informational purposes only "
                "and does not constitute legal advice. For any real legal matter, "
                "please consult a licensed lawyer or legal professional in Pakistan."
            )

        return {
            "answer": answer,
            "sources": [
                {
                    "citation": c.citation,
                    "source": c.source,
                    "page": c.page,
                    "confidence": c.confidence_pct,
                    "confidence_label": c.confidence_label,
                    "text": c.text,
                }
                for c in chunks
            ],
            "context_used": context_str,
        }
