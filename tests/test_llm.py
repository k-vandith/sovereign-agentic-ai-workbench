from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.llm.base as llm_base
from src.llm import DemoLLM, OllamaLLM, OpenAICompatibleLLM, OfflineBlockedError


def test_get_llm_uses_configured_local_ollama_in_offline_mode(monkeypatch):
    settings = SimpleNamespace(
        llm_backend="ollama",
        offline_mode=True,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2:1b",
        ollama_timeout=120,
        openai_compatible_base_url="",
        openai_compatible_api_key="",
        openai_compatible_model="",
    )
    monkeypatch.setattr(llm_base, "get_settings", lambda: settings)

    assert isinstance(llm_base.get_llm(), OllamaLLM)


def test_offline_mode_blocks_non_local_ollama_endpoint(monkeypatch):
    settings = SimpleNamespace(
        llm_backend="ollama",
        offline_mode=True,
        ollama_base_url="https://model.example",
        ollama_model="llama3.2:1b",
        ollama_timeout=120,
        openai_compatible_base_url="",
        openai_compatible_api_key="",
        openai_compatible_model="",
    )
    monkeypatch.setattr(llm_base, "get_settings", lambda: settings)

    assert isinstance(llm_base.get_llm(), DemoLLM)


def test_explicit_remote_api_call_is_blocked_in_offline_mode(monkeypatch):
    settings = SimpleNamespace(offline_mode=True)
    monkeypatch.setattr(llm_base, "get_settings", lambda: settings)
    called = False

    class NoNetworkClient:
        def __init__(self, *args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("network client should not be constructed")

    monkeypatch.setattr(llm_base.httpx, "Client", NoNetworkClient)
    llm = OpenAICompatibleLLM(
        base_url="https://api.example",
        api_key="test-secret",
        model="test-model",
    )
    with pytest.raises(OfflineBlockedError, match="OFFLINE_MODE=true"):
        llm.generate("question")
    assert called is False


def test_empty_vision_response_falls_back_to_local_image_metadata(tmp_path, monkeypatch):
    from PIL import Image

    image_path = tmp_path / "tiny.png"
    Image.new("RGB", (2, 2)).save(image_path)
    settings = SimpleNamespace(
        ollama_vision_model="llava",
        ollama_timeout=2,
        offline_mode=True,
    )
    monkeypatch.setattr(llm_base, "get_settings", lambda: settings)

    class FakeOllama:
        base_url = "http://localhost:11434"

        def __init__(self, model=None):
            self.model = model

        def is_available(self):
            return True

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"message": {"content": "   "}}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            assert kwargs["trust_env"] is False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(llm_base, "OllamaLLM", FakeOllama)
    monkeypatch.setattr(llm_base.httpx, "Client", FakeClient)

    result = llm_base.describe_image_with_vision(str(image_path))

    assert "Content: local metadata analysis only." in result
