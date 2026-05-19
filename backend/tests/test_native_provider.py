"""Tests for NativeMemoryProvider wrapping MemoryStore."""

import json
from pathlib import Path

from deerflow.agents.memory.store import MemoryStore
from deerflow.memory_providers.native import NativeMemoryProvider


def _make_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(
        memory_dir=tmp_path,
        memory_char_limit=2200,
        user_char_limit=1375,
    )
    store.load_from_disk()
    return store


class TestNativeProvider:
    def test_name(self, tmp_path):
        store = _make_store(tmp_path)
        p = NativeMemoryProvider(store=store)
        assert p.name == "native"

    def test_is_available_always_true(self, tmp_path):
        store = _make_store(tmp_path)
        p = NativeMemoryProvider(store=store)
        assert p.is_available() is True

    def test_system_prompt_block_returns_snapshot(self, tmp_path):
        store = _make_store(tmp_path)
        store.add("memory", "test entry")
        p = NativeMemoryProvider(store=store)
        block = p.system_prompt_block()
        assert "test entry" in block

    def test_system_prompt_block_empty_when_no_entries(self, tmp_path):
        store = _make_store(tmp_path)
        p = NativeMemoryProvider(store=store)
        block = p.system_prompt_block()
        assert block == ""

    def test_get_tool_schemas_returns_memory_schema(self, tmp_path):
        store = _make_store(tmp_path)
        p = NativeMemoryProvider(store=store)
        schemas = p.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "memory"

    def test_handle_tool_call_add(self, tmp_path):
        store = _make_store(tmp_path)
        p = NativeMemoryProvider(store=store)
        result = p.handle_tool_call(
            "memory",
            {
                "action": "add",
                "target": "memory",
                "content": "hello world",
            },
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_handle_tool_call_unknown_raises(self, tmp_path):
        store = _make_store(tmp_path)
        p = NativeMemoryProvider(store=store)
        import pytest

        with pytest.raises(ValueError, match="Unknown tool"):
            p.handle_tool_call("nonexistent", {})

    def test_prefetch_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        p = NativeMemoryProvider(store=store)
        assert p.prefetch("test") == ""

    def test_sync_turn_is_noop(self, tmp_path):
        store = _make_store(tmp_path)
        p = NativeMemoryProvider(store=store)
        p.sync_turn("user", "assistant")  # no raise

    def test_initialize_and_shutdown(self, tmp_path):
        store = _make_store(tmp_path)
        p = NativeMemoryProvider(store=store)
        p.initialize("session_123")
        p.shutdown()  # no raise
