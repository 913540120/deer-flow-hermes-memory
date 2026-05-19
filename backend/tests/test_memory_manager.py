"""Tests for MemoryManager orchestrator."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deerflow.agents.memory.store import MemoryStore
from deerflow.memory_providers.manager import MemoryManager
from deerflow.memory_providers.native import NativeMemoryProvider
from deerflow.memory_providers.provider import MemoryProvider


def _make_native(tmp_path: Path) -> NativeMemoryProvider:
    store = MemoryStore(memory_dir=tmp_path, memory_char_limit=2200, user_char_limit=1375)
    store.load_from_disk()
    return NativeMemoryProvider(store=store)


class FakeExternalProvider(MemoryProvider):
    """Fake external provider for testing."""

    def __init__(self, *, available: bool = True):
        self._available = available
        self._initialized = False
        self._shutdown_called = False
        self._synced_turns: list[tuple[str, str]] = []
        self._prefetch_results: str = ""
        self._tools: list[dict] = []
        self._tool_responses: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return self._available

    def initialize(self, session_id: str, **kwargs) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._shutdown_called = True

    def system_prompt_block(self) -> str:
        return "External memory active."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return self._prefetch_results

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        self._synced_turns.append((user_content, assistant_content))

    def get_tool_schemas(self) -> list[dict]:
        return self._tools

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        return self._tool_responses.get(tool_name, '{"ok": true}')


class TestMemoryManagerBasic:
    def test_native_only(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        assert mgr.native is native
        assert mgr.external is None

    def test_add_external_provider(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        mgr.add_provider(ext)
        assert mgr.external is ext

    def test_add_second_external_raises(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        mgr.add_provider(FakeExternalProvider())
        with pytest.raises(ValueError, match="Already have external provider"):
            mgr.add_provider(FakeExternalProvider())


class TestMemoryManagerLifecycle:
    def test_initialize_all(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        mgr.add_provider(ext)
        mgr.initialize_all("session_1")
        assert ext._initialized is True

    def test_initialize_all_external_failure_logged(self, tmp_path, caplog):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        ext.initialize = MagicMock(side_effect=RuntimeError("init fail"))
        mgr.add_provider(ext)
        with caplog.at_level(logging.WARNING):
            mgr.initialize_all("session_1")  # should not raise
        assert "init fail" in caplog.text

    def test_shutdown_all(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        mgr.add_provider(ext)
        mgr.shutdown_all()
        assert ext._shutdown_called is True

    def test_shutdown_all_external_failure_logged(self, tmp_path, caplog):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        ext.shutdown = MagicMock(side_effect=RuntimeError("shutdown fail"))
        mgr.add_provider(ext)
        with caplog.at_level(logging.WARNING):
            mgr.shutdown_all()  # should not raise
        assert "shutdown fail" in caplog.text


class TestMemoryManagerPrompt:
    def test_build_system_prompt_native_only(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        result = mgr.build_system_prompt()
        assert result == ""

    def test_build_system_prompt_with_native_content(self, tmp_path):
        native = _make_native(tmp_path)
        native._store.add("memory", "test fact")
        mgr = MemoryManager(native)
        result = mgr.build_system_prompt()
        assert "test fact" in result

    def test_build_system_prompt_with_external(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        mgr.add_provider(ext)
        result = mgr.build_system_prompt()
        assert "External memory active." in result

    def test_build_external_prompt_returns_external_block(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        mgr.add_provider(ext)
        assert mgr.build_external_prompt() == "External memory active."

    def test_build_external_prompt_no_external(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        assert mgr.build_external_prompt() == ""


class TestMemoryManagerPerTurn:
    def test_sync_all_with_external(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        mgr.add_provider(ext)
        mgr.sync_all("hello", "world", session_id="s1")
        assert ext._synced_turns == [("hello", "world")]

    def test_sync_all_external_failure_logged(self, tmp_path, caplog):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        ext.sync_turn = MagicMock(side_effect=RuntimeError("sync fail"))
        mgr.add_provider(ext)
        with caplog.at_level(logging.WARNING):
            mgr.sync_all("hello", "world")  # should not raise
        assert "sync fail" in caplog.text

    def test_prefetch_all(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        ext._prefetch_results = "some context"
        mgr.add_provider(ext)
        result = mgr.prefetch_all("test query")
        assert result == "some context"

    def test_prefetch_all_failure_returns_empty(self, tmp_path, caplog):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        ext.prefetch = MagicMock(side_effect=RuntimeError("prefetch fail"))
        mgr.add_provider(ext)
        with caplog.at_level(logging.WARNING):
            result = mgr.prefetch_all("test")
        assert result == ""
        assert "prefetch fail" in caplog.text


class TestMemoryManagerToolRouting:
    def test_get_all_tool_schemas_native_only(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        schemas = mgr.get_all_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "memory"

    def test_get_all_tool_schemas_with_external(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        ext._tools = [{"name": "ext_search", "description": "Search", "parameters": {}}]
        mgr.add_provider(ext)
        mgr.initialize_all("session_1")
        schemas = mgr.get_all_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "memory" in names
        assert "ext_search" in names

    def test_handle_tool_call_routes_to_native(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        mgr.initialize_all("session_1")
        result = mgr.handle_tool_call(
            "memory",
            {
                "action": "add",
                "target": "memory",
                "content": "hello",
            },
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_handle_tool_call_routes_to_external(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        ext = FakeExternalProvider()
        ext._tools = [{"name": "ext_search", "description": "Search", "parameters": {}}]
        ext._tool_responses = {"ext_search": '{"results": []}'}
        mgr.add_provider(ext)
        mgr.initialize_all("session_1")
        result = mgr.handle_tool_call("ext_search", {"query": "test"})
        assert result == '{"results": []}'

    def test_handle_tool_call_unknown_raises(self, tmp_path):
        native = _make_native(tmp_path)
        mgr = MemoryManager(native)
        mgr.initialize_all("session_1")
        with pytest.raises(ValueError, match="Unknown tool"):
            mgr.handle_tool_call("nonexistent", {})
