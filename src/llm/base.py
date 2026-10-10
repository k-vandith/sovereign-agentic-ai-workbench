"""Language generation adapters for API-powered answers and preview mode."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


class DemoLLM(BaseLLM):
    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        context_block = ""
        if "Context:" in prompt:
            try:
                context_block = prompt.split("Context:")[1].split("Question:")[0].strip()
            except Exception:
                context_block = ""
        question = prompt.split("Question:")[-1].strip() if "Question:" in prompt else prompt
        if context_block:
            return (
                "[Demo Mode Response]\n\nBased on the retrieved local knowledge base:\n\n"
                f"{context_block[:800]}\n\n---\nPreview response. Configure an API in .env for stronger answers."
            )
        return (
            f"[Demo Mode Response]\n\nQuery: {question[:200]}\n\n"
            "No relevant documents retrieved. Upload documents or configure an API for stronger answers."
        )


class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: int | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                return client.get(f"{self.base_url}/api/tags").status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"num_predict": max_tokens, "temperature": temperature}}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "") or data.get("response", "")
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.openai_compatible_base_url).rstrip("/")
        self.api_key = api_key or settings.openai_compatible_api_key
        self.model = model or settings.openai_compatible_model

    def is_available(self) -> bool:
        return bool(self.base_url and self.model)

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        if not self.is_available():
            raise RuntimeError("OpenAI-compatible backend is not configured.")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        api_root = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{api_root}/chat/completions",
                            json={"model": self.model, "messages": messages,
                                  "max_tokens": max_tokens, "temperature": temperature},
                            headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


class OfflineBlockedError(RuntimeError):
    pass


def get_llm() -> BaseLLM:
    """Choose the configured API when explicitly enabled; otherwise use safe preview mode."""
    settings = get_settings()
    if settings.llm_backend != "openai_compatible":
        return DemoLLM()
    if settings.offline_mode:
        logger.warning("offline_mode is enabled; using preview mode instead of a remote API")
        return DemoLLM()
    llm = OpenAICompatibleLLM()
    if llm.is_available():
        return llm
    logger.warning("API settings are incomplete; using preview mode")
    return DemoLLM()


def describe_image_with_vision(image_path: str, prompt: str = "Describe this industrial image.") -> str:
    """Ollama vision (llava) when available; else Pillow metadata fallback."""
    settings = get_settings()
    path = Path(image_path)
    if not path.exists():
        return f"Image not found: {image_path}"
    try:
        import base64
        llm = OllamaLLM(model=settings.ollama_vision_model)
        if llm.is_available():
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            payload = {"model": settings.ollama_vision_model, "messages": [
                {"role": "user", "content": prompt, "images": [b64]}
            ], "stream": False}
            with httpx.Client(timeout=settings.ollama_timeout) as client:
                r = client.post(f"{llm.base_url}/api/chat", json=payload)
                if r.status_code == 200:
                    return r.json().get("message", {}).get("content", "") or "Empty vision response"
    except Exception as exc:
        logger.info("Vision model unavailable, falling back: %s", exc)
    from src.agent.tools import ToolRegistry
    return ToolRegistry().image_analysis(str(path))
