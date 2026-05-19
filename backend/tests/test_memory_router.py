from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.routers import memory


@pytest.fixture
def memory_app(tmp_path):
    """Create a FastAPI app with the memory router, patched to use tmp_path."""
    app = FastAPI()
    app.include_router(memory.router)

    config = SimpleNamespace(
        enabled=True,
        injection_enabled=True,
        storage_path="",
        memory_char_limit=2200,
        user_char_limit=1375,
    )
    fake_paths = SimpleNamespace(user_dir=lambda uid: tmp_path / "users" / uid)

    with (
        patch("app.gateway.routers.memory.get_memory_config", return_value=config),
        patch("app.gateway.routers.memory.get_paths", return_value=fake_paths),
        patch("app.gateway.routers.memory.get_effective_user_id", return_value="test-user"),
    ):
        yield app


def test_get_memory_returns_empty_entries(memory_app) -> None:
    with TestClient(memory_app) as client:
        response = client.get("/api/memory")

    assert response.status_code == 200
    data = response.json()
    assert "memory" in data
    assert "user" in data
    assert data["memory"]["entries"] == []
    assert data["user"]["entries"] == []


def test_add_memory_entry(memory_app) -> None:
    with TestClient(memory_app) as client:
        # Add a memory entry
        response = client.post(
            "/api/memory/entries",
            json={"action": "add", "target": "memory", "content": "I know this test works"},
        )

    assert response.status_code == 200
    data = response.json()
    assert any("I know this test works" in e for e in data["memory"]["entries"])


def test_replace_memory_entry(memory_app) -> None:
    with TestClient(memory_app) as client:
        # Add an entry first
        client.post(
            "/api/memory/entries",
            json={"action": "add", "target": "memory", "content": "old content"},
        )
        # Replace it
        response = client.post(
            "/api/memory/entries",
            json={"action": "replace", "target": "memory", "old_text": "old content", "content": "new content"},
        )

    assert response.status_code == 200
    data = response.json()
    assert any("new content" in e for e in data["memory"]["entries"])
    assert not any("old content" in e for e in data["memory"]["entries"])


def test_remove_memory_entry(memory_app) -> None:
    with TestClient(memory_app) as client:
        # Add an entry first
        client.post(
            "/api/memory/entries",
            json={"action": "add", "target": "memory", "content": "to be removed"},
        )
        # Remove it
        response = client.post(
            "/api/memory/entries",
            json={"action": "remove", "target": "memory", "old_text": "to be removed"},
        )

    assert response.status_code == 200
    data = response.json()
    assert not any("to be removed" in e for e in data["memory"]["entries"])


def test_clear_memory(memory_app) -> None:
    with TestClient(memory_app) as client:
        # Add entries first
        client.post(
            "/api/memory/entries",
            json={"action": "add", "target": "memory", "content": "some content"},
        )
        # Clear all
        response = client.delete("/api/memory")

    assert response.status_code == 200
    data = response.json()
    assert data["memory"]["entries"] == []
    assert data["user"]["entries"] == []


def test_get_memory_config(memory_app) -> None:
    with TestClient(memory_app) as client:
        response = client.get("/api/memory/config")

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["injection_enabled"] is True
    assert data["memory_char_limit"] == 2200
    assert data["user_char_limit"] == 1375


def test_get_memory_status(memory_app) -> None:
    with TestClient(memory_app) as client:
        response = client.get("/api/memory/status")

    assert response.status_code == 200
    data = response.json()
    assert "config" in data
    assert "data" in data
    assert data["config"]["enabled"] is True


def test_add_entry_rejects_missing_content(memory_app) -> None:
    with TestClient(memory_app) as client:
        response = client.post(
            "/api/memory/entries",
            json={"action": "add", "target": "memory"},
        )

    assert response.status_code == 400


def test_unknown_action_rejected(memory_app) -> None:
    with TestClient(memory_app) as client:
        response = client.post(
            "/api/memory/entries",
            json={"action": "unknown", "target": "memory", "content": "test"},
        )

    assert response.status_code == 400


def test_user_target_entries(memory_app) -> None:
    with TestClient(memory_app) as client:
        response = client.post(
            "/api/memory/entries",
            json={"action": "add", "target": "user", "content": "User prefers Python"},
        )

    assert response.status_code == 200
    data = response.json()
    assert any("User prefers Python" in e for e in data["user"]["entries"])
