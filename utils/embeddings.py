"""
Lightweight ONNX-based embeddings (via ChromaDB's built-in model) wrapped
as a LangChain Embeddings class. Avoids torch/sentence-transformers entirely,
so it fits comfortably in Render's 512MB free-tier RAM and needs no
external API calls.
"""

from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


class LightweightEmbeddings(Embeddings):
    def __init__(self) -> None:
        self._ef = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0]