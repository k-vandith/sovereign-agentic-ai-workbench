"""Configuration management using pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_backend: Literal["ollama", "demo", "openai_compatible"] = "demo"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout: int = 120

    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_model: str = ""

    embedding_model: str = "all-MiniLM-L6-v2"

    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    vectorstore_dir: Path = Path("data/vectorstore")
    conversation_dir: Path = Path("data/conversations")

    top_k: int = Field(default=5, ge=1, le=50)
    relevance_threshold: float = Field(default=0.15, ge=-1.0, le=1.0)
    conversation_history_tokens: int = Field(default=2000, ge=0, le=20000)
    chunk_size: int = Field(default=500, ge=1, le=10000)
    chunk_overlap: int = Field(default=50, ge=0, le=9999)

    @model_validator(mode="after")
    def validate_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    streamlit_port: int = 8501
    api_token: str = ""

    log_level: str = "INFO"

    offline_mode: bool = True
    ollama_vision_model: str = "llava"
    default_role: str = "operator"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.upload_dir,
            self.vectorstore_dir,
            self.conversation_dir,
            self.data_dir / "audit",
        ):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
