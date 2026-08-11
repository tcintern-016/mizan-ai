"""
Central configuration management for Pakistan Law Assistant.
Loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # App
    app_name: str = Field(default="PAKISTAN LAW ASSISTANT", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")

    # Groq
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="pakistan_law", env="CHROMA_COLLECTION_NAME")

    # Embedding
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")

    # Retrieval
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    similarity_threshold: float = Field(default=0.25, env="SIMILARITY_THRESHOLD")

    # Frontend
    api_base_url: str = Field(default="http://localhost:8000/api/v1", env="API_BASE_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def data_raw_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "data" / "raw"


settings = Settings()
