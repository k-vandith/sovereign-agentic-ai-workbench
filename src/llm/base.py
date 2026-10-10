"""Language generation adapters for API-powered answers and preview mode."""
from __future__ import annotations

import ipaddress
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)


def _is_loopback_url(url: str) -> bool:
    """Return True only for localhost/loopback endpoints, suitable for offline mode."""
    value = str(url or "").strip()
    if not value:
        return False
    try:
        parsed = urlsplit(value if "://" in value else f"http://{value}")
        host = parsed.hostname
        if not host:
            return False
        if host.lower() == "localhost":
            return True
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
        settings = get_settings()
        if settings.offline_mode and not _is_loopback_url(self.base_url):
            return False
        try:
            with httpx.Client(timeout=5.0, trust_env=not settings.offline_mode) as client:
                return client.get(f"{self.base_url}/api/tags").status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        settings = get_settings()
        if settings.offline_mode and not _is_loopback_url(self.base_url):
            raise OfflineBlockedError("Remote Ollama requests are disabled while OFFLINE_MODE=true.")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"num_predict": max_tokens, "temperature": temperature}}
        try:
            with httpx.Client(timeout=self.timeout, trust_env=not settings.offline_mode) as client:
                r = client.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Ollama provider returned HTTP %s.", exc.response.status_code)
            raise RuntimeError(f"Ollama request failed with HTTP {exc.response.status_code}.") from exc
        except httpx.RequestError as exc:
            logger.warning("Ollama request failed (%s).", type(exc).__name__)
            raise RuntimeError("Ollama request failed due to a connection or timeout error.") from exc
        try:
            data = r.json()
            message = data.get("message") if isinstance(data, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not content and isinstance(data, dict):
                content = data.get("response")
        except (ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError("Ollama returned a malformed response.") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty or malformed response.")
        return content


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.openai_compatible_base_url).rstrip("/")
        self.api_key = api_key or settings.openai_compatible_api_key
        self.model = model or settings.openai_compatible_model

    def is_available(self) -> bool:
        return bool(self.base_url and self.model)

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        if get_settings().offline_mode:
            raise OfflineBlockedError("Remote API requests are disabled while OFFLINE_MODE=true.")
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
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{api_root}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("OpenAI-compatible provider returned HTTP %s.", exc.response.status_code)
            raise RuntimeError(
                f"OpenAI-compatible request failed with HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("OpenAI-compatible request failed (%s).", type(exc).__name__)
            raise RuntimeError(
                "OpenAI-compatible request failed due to a connection or timeout error."
            ) from exc
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI-compatible provider returned a malformed response.") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI-compatible provider returned an empty answer.")
        return content


class OfflineBlockedError(RuntimeError):
    pass


def get_llm() -> BaseLLM:
    """Select the configured local or remote backend without accidental network calls."""
    settings = get_settings()
    if settings.llm_backend == "demo":
        return DemoLLM()
    if settings.llm_backend == "ollama":
        if settings.offline_mode and not _is_loopback_url(settings.ollama_base_url):
            logger.warning("offline_mode blocks the configured non-local Ollama endpoint; using preview mode")
            return DemoLLM()
        return OllamaLLM()
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
            with httpx.Client(timeout=settings.ollama_timeout, trust_env=not settings.offline_mode) as client:
                r = client.post(f"{llm.base_url}/api/chat", json=payload)
                if r.status_code == 200:
                    data = r.json()
                    message = data.get("message") if isinstance(data, dict) else None
                    content = message.get("content") if isinstance(message, dict) else None
                    if not content and isinstance(data, dict):
                        content = data.get("response")
                    if isinstance(content, str) and content.strip():
                        return content
                    logger.info("Vision model returned empty content; using local metadata fallback.")
    except Exception as exc:
        logger.info("Vision model unavailable, falling back to local image metadata (%s).", type(exc).__name__)

    # This helper is called with an explicit image path; keep its metadata fallback
    # separate from the agent-exposed tool, whose paths are restricted to data roots.
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            mode, image_format = image.mode, image.format or path.suffix
        return (
            f"Image file: {path.name}\\nFormat: {image_format}, mode: {mode}, "
            f"size: {width}x{height} px\\n"
            "Content: local metadata analysis only. Semantic image understanding is not enabled in this build."
        )
    except Exception:
        return "Vision model unavailable and local image metadata could not be read."
