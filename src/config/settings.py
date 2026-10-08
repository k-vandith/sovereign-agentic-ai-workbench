"""Configuration management using pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
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

    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create required directories if they do not exist."""
        for d in (
            self.data_dir,
            self.upload_dir,
            self.vectorstore_dir,
            self.conversation_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
