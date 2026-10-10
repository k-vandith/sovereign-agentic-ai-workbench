from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.main as api


TEST_API_TOKEN = "test-only-token-for-local-api-tests"


def _auth(monkeypatch):
    monkeypatch.setattr(api.settings, "api_token", TEST_API_TOKEN)
    return {"Authorization": f"Bearer {TEST_API_TOKEN}"}


def test_ingest_uses_a_temporary_file_and_cleans_it(monkeypatch):
    headers = _auth(monkeypatch)
    captured: dict[str, object] = {}

    def ingest_file(path: Path):
        captured["path"] = path
        captured["bytes"] = path.read_bytes()
        return {"file": path.name, "chunks_added": 1, "total_chunks": 1}

    monkeypatch.setattr(api, "agent", SimpleNamespace(ingest_file=ingest_file))
    with TestClient(api.app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("manual.txt", b"local procedure", "text/plain")},
            headers=headers,
        )

    assert response.status_code == 200
    assert captured["bytes"] == b"local procedure"
    assert not captured["path"].exists()


def test_ingest_rejects_oversized_file_before_indexing(monkeypatch):
    headers = _auth(monkeypatch)
    called = False

    def ingest_file(_path: Path):
        nonlocal called
        called = True
        return {"chunks_added": 1}

    monkeypatch.setattr(api, "agent", SimpleNamespace(ingest_file=ingest_file))
    monkeypatch.setattr(api, "MAX_FILE_BYTES", 4)
    with TestClient(api.app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("manual.txt", b"12345", "text/plain")},
            headers=headers,
        )

    assert response.status_code == 413
    assert called is False


def test_query_rejects_blank_questions_and_invalid_top_k(monkeypatch):
    headers = _auth(monkeypatch)
    called = False

    def query(**_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(api, "agent", SimpleNamespace(query=query))
    with TestClient(api.app) as client:
        blank = client.post("/query", json={"question": "   "}, headers=headers)
        invalid_k = client.post("/query", json={"question": "question", "top_k": 0}, headers=headers)
        boolean_k = client.post("/query", json={"question": "question", "top_k": True}, headers=headers)

    assert blank.status_code == 422
    assert invalid_k.status_code == 422
    assert boolean_k.status_code == 422
    assert called is False


def test_query_hides_unexpected_backend_error_details(monkeypatch):
    headers = _auth(monkeypatch)
    def query(**_kwargs):
        raise RuntimeError("provider internals and secret=never-expose")

    monkeypatch.setattr(api, "agent", SimpleNamespace(query=query))
    with TestClient(api.app) as client:
        response = client.post("/query", json={"question": "question"}, headers=headers)

    assert response.status_code == 500
    assert "secret" not in response.text
    assert "Check the backend configuration" in response.text


def test_query_maps_direct_validation_errors_to_bad_request(monkeypatch):
    headers = _auth(monkeypatch)
    def query(**_kwargs):
        raise ValueError("Question must not be blank.")

    monkeypatch.setattr(api, "agent", SimpleNamespace(query=query))
    with TestClient(api.app) as client:
        response = client.post("/query", json={"question": "question"}, headers=headers)

    assert response.status_code == 400
    assert "Question must not be blank" in response.text


def test_clear_knowledge_base_returns_forbidden_for_unauthorized_role(monkeypatch):
    headers = _auth(monkeypatch)
    def clear_knowledge_base():
        raise PermissionError("internal role details")

    monkeypatch.setattr(api, "agent", SimpleNamespace(clear_knowledge_base=clear_knowledge_base))
    with TestClient(api.app) as client:
        response = client.delete("/knowledge-base", headers=headers)

    assert response.status_code == 403
    assert "internal role details" not in response.text


def test_cors_only_allows_configured_local_ui_origins():
    allowed_origin = f"http://localhost:{api.settings.streamlit_port}"
    headers = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    with TestClient(api.app) as client:
        allowed = client.options(
            "/query",
            headers={**headers, "Origin": allowed_origin},
        )
        rejected = client.options(
            "/query",
            headers={**headers, "Origin": "https://attacker.example"},
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == allowed_origin
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers


def test_ingest_rejects_simple_cross_origin_post_before_indexing(monkeypatch):
    auth_headers = _auth(monkeypatch)
    called = False

    def ingest_file(_path: Path):
        nonlocal called
        called = True
        return {"chunks_added": 1}

    monkeypatch.setattr(api, "agent", SimpleNamespace(ingest_file=ingest_file))
    with TestClient(api.app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("manual.txt", b"safe content", "text/plain")},
            headers={**auth_headers, "Origin": "https://attacker.example"},
        )

    assert response.status_code == 403
    assert "Cross-origin" in response.text
    assert called is False



@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/ingest", {"files": {"file": ("manual.txt", b"data", "text/plain")}}),
        ("post", "/query", {"json": {"question": "What is documented?"}}),
        ("delete", "/knowledge-base", {}),
    ],
)
def test_protected_routes_require_a_valid_bearer_token(monkeypatch, method, path, kwargs):
    _auth(monkeypatch)
    with TestClient(api.app) as client:
        response = getattr(client, method)(path, **kwargs)

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_protected_routes_fail_closed_when_api_token_is_unconfigured(monkeypatch):
    monkeypatch.setattr(api.settings, "api_token", "")
    with TestClient(api.app) as client:
        response = client.post("/query", json={"question": "What is documented?"})

    assert response.status_code == 503
    assert "API_TOKEN" in response.text


def test_cross_origin_query_and_delete_are_rejected_even_with_valid_token(monkeypatch):
    headers = {**_auth(monkeypatch), "Origin": "https://attacker.example"}
    with TestClient(api.app) as client:
        query_response = client.post("/query", json={"question": "question"}, headers=headers)
        delete_response = client.delete("/knowledge-base", headers=headers)

    assert query_response.status_code == 403
    assert delete_response.status_code == 403


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/conversations", "get"),
        ("/conversations/private-id", "get"),
        ("/stats", "get"),
    ],
)
def test_private_read_routes_require_a_valid_bearer_token(monkeypatch, path, method):
    _auth(monkeypatch)
    with TestClient(api.app) as client:
        response = getattr(client, method)(path)

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
