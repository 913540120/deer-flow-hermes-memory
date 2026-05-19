"""Tests for MemoryManager integration with lead agent and prompt."""

import json
from pathlib import Path

from deerflow.agents.memory.store import MemoryStore
from deerflow.memory_providers.manager import MemoryManager
from deerflow.memory_providers.native import NativeMemoryProvider
from deerflow.memory_providers.provider import MemoryProvider


def _make_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(memory_dir=tmp_path, memory_char_limit=2200, user_char_limit=1375)
    store.load_from_disk()
    return store


class TestAgentIntegration:
    def test_memory_manager_created_when_memory_enabled(self, tmp_path):
        store = _make_store(tmp_path)
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)
        mgr.initialize_all("thread_123")
        assert mgr._initialized is True

    def test_external_provider_not_loaded_when_disabled(self, tmp_path):
        store = _make_store(tmp_path)
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)
        assert mgr.external is None

    def test_external_provider_registered_when_available(self, tmp_path):
        store = _make_store(tmp_path)
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)

        class FakeProvider(MemoryProvider):
            @property
            def name(self):
                return "fake"

            def is_available(self):
                return True

        ext = FakeProvider()
        mgr.add_provider(ext)
        mgr.initialize_all("thread_123")
        assert mgr.external is ext

    def test_native_tool_still_works_with_manager(self, tmp_path):
        store = _make_store(tmp_path)
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)
        mgr.initialize_all("thread_123")

        schemas = mgr.get_all_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "memory" in names

        result = mgr.handle_tool_call(
            "memory",
            {
                "action": "add",
                "target": "memory",
                "content": "integration test",
            },
        )
        parsed = json.loads(result)
        assert parsed["success"] is True


class TestPromptIntegration:
    def test_external_prompt_appended_to_memory_context(self, tmp_path):
        store = _make_store(tmp_path)
        store.add("memory", "native fact")
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)

        assert mgr.build_external_prompt() == ""

        class FakeProvider(MemoryProvider):
            @property
            def name(self):
                return "fake"

            def is_available(self):
                return True

            def system_prompt_block(self):
                return "External memory active."

        ext = FakeProvider()
        mgr.add_provider(ext)

        ext_block = mgr.build_external_prompt()
        assert ext_block == "External memory active."

        full = mgr.build_system_prompt()
        assert "native fact" in full
        assert "External memory active." in full
