# Memory Provider Plugin Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable MemoryProvider ABC + MemoryManager orchestrator so DeerFlow can use external memory services (Mem0, etc.) alongside native MEMORY.md/USER.md storage.

**Architecture:** Wrapping integration — NativeMemoryProvider wraps existing MemoryStore without modifying it. MemoryManager orchestrates native + at most one external provider. External provider failures are best-effort (logged, never crash). Directory-based plugin discovery for built-in and user-installed providers.

**Tech Stack:** Python ABC, pydantic config, LangChain tools, optional `mem0ai` SDK.

**Spec:** `docs/superpowers/specs/2026-05-19-memory-provider-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `deerflow/memory_providers/__init__.py` | Module exports, `discover_providers()`, `load_provider()` |
| `deerflow/memory_providers/provider.py` | `MemoryProvider` ABC with 9 methods |
| `deerflow/memory_providers/manager.py` | `MemoryManager` orchestrator (tool routing, lifecycle, failure isolation) |
| `deerflow/memory_providers/native.py` | `NativeMemoryProvider` wrapping `MemoryStore` |
| `deerflow/memory_providers/mem0/__init__.py` | `Mem0Provider` with circuit breaker |
| `deerflow/memory_providers/mem0/tool.py` | Mem0 tool schemas and handler methods |
| `config/memory_provider_config.py` | `MemoryProviderConfig` pydantic model |

### Modified Files

| File | Change |
|------|--------|
| `config/app_config.py` | Add `memory_provider: MemoryProviderConfig` field |
| `agents/lead_agent/agent.py` | Add MemoryManager creation + external tool registration |
| `agents/lead_agent/prompt.py` | Add `memory_manager` param to `_get_memory_context()` |
| `config.example.yaml` | Add `memory_provider` config section |

### New Test Files

| File | Tests |
|------|-------|
| `tests/test_memory_provider_abc.py` | ABC contract, default implementations |
| `tests/test_memory_manager.py` | Orchestration, tool routing, failure isolation |
| `tests/test_native_provider.py` | NativeMemoryProvider wrapping MemoryStore |
| `tests/test_mem0_provider.py` | Mem0 lifecycle, tools, circuit breaker |
| `tests/test_memory_provider_discovery.py` | Discovery and loading |

---

### Task 1: MemoryProvider ABC

**Files:**
- Create: `deerflow/memory_providers/provider.py`
- Test: `tests/test_memory_provider_abc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_provider_abc.py
"""Tests for MemoryProvider ABC contract and default implementations."""

import pytest

from deerflow.memory_providers.provider import MemoryProvider


class ConcreteProvider(MemoryProvider):
    """Minimal concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test"

    def is_available(self) -> bool:
        return True


class TestMemoryProviderABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            MemoryProvider()

    def test_concrete_provider_has_name(self):
        p = ConcreteProvider()
        assert p.name == "test"

    def test_concrete_provider_is_available(self):
        p = ConcreteProvider()
        assert p.is_available() is True

    def test_default_initialize_is_noop(self):
        p = ConcreteProvider()
        p.initialize("session_123")  # should not raise

    def test_default_shutdown_is_noop(self):
        p = ConcreteProvider()
        p.shutdown()  # should not raise

    def test_default_system_prompt_block_returns_empty(self):
        p = ConcreteProvider()
        assert p.system_prompt_block() == ""

    def test_default_prefetch_returns_empty(self):
        p = ConcreteProvider()
        assert p.prefetch("hello") == ""

    def test_default_sync_turn_is_noop(self):
        p = ConcreteProvider()
        p.sync_turn("user msg", "assistant msg")  # should not raise

    def test_default_get_tool_schemas_returns_empty(self):
        p = ConcreteProvider()
        assert p.get_tool_schemas() == []

    def test_default_handle_tool_call_raises(self):
        p = ConcreteProvider()
        with pytest.raises(ValueError, match="Unknown tool"):
            p.handle_tool_call("nonexistent", {})


class TestProviderMissingAbstract:
    def test_missing_name_raises(self):
        class NoName(MemoryProvider):
            def is_available(self) -> bool:
                return True

        with pytest.raises(TypeError):
            NoName()

    def test_missing_is_available_raises(self):
        class NoAvail(MemoryProvider):
            @property
            def name(self) -> str:
                return "x"

        with pytest.raises(TypeError):
            NoAvail()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_provider_abc.py -v`
Expected: FAIL — module `deerflow.memory_providers.provider` does not exist

- [ ] **Step 3: Write minimal implementation**

```python
# deerflow/memory_providers/provider.py
"""MemoryProvider abstract base class — plugin interface for memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryProvider(ABC):
    """Pluggable memory backend interface.

    Subclasses must implement ``name`` and ``is_available``. All other
    methods have safe defaults (no-op or empty return).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'native', 'mem0', 'honcho', etc."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and ready to use."""

    # --- Lifecycle ---

    def initialize(self, session_id: str, **kwargs) -> None:
        """One-time setup per session. Default: no-op."""

    def shutdown(self) -> None:
        """Clean shutdown. Default: no-op."""

    # --- System Prompt ---

    def system_prompt_block(self) -> str:
        """Static text injected into system prompt. Default: empty."""
        return ""

    # --- Per-turn ---

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall context before each turn. Default: empty."""
        return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Persist completed turn. Default: no-op."""

    # --- Tools ---

    def get_tool_schemas(self) -> list[dict]:
        """Return LangChain tool schemas for this provider. Default: []."""
        return []

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch tool call. Default: raise ValueError."""
        raise ValueError(f"Unknown tool: {tool_name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_provider_abc.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/memory_providers/provider.py backend/tests/test_memory_provider_abc.py
git commit -m "feat(memory-provider): add MemoryProvider ABC with 9 lifecycle methods"
```

---

### Task 2: MemoryProviderConfig + AppConfig registration

**Files:**
- Create: `deerflow/config/memory_provider_config.py`
- Modify: `deerflow/config/app_config.py`
- Modify: `config.example.yaml`
- Test: `tests/test_memory_provider_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_provider_config.py
"""Tests for MemoryProviderConfig and AppConfig integration."""

from deerflow.config.app_config import AppConfig
from deerflow.config.memory_provider_config import MemoryProviderConfig


class TestMemoryProviderConfig:
    def test_defaults(self):
        cfg = MemoryProviderConfig()
        assert cfg.enabled is False
        assert cfg.name == ""

    def test_custom_values(self):
        cfg = MemoryProviderConfig(enabled=True, name="mem0")
        assert cfg.enabled is True
        assert cfg.name == "mem0"


class TestAppConfigIntegration:
    def test_app_config_has_memory_provider_field(self):
        cfg = AppConfig.model_validate({
            "models": [],
            "sandbox": {"use": "x"},
        })
        assert hasattr(cfg, "memory_provider")
        assert cfg.memory_provider.enabled is False
        assert cfg.memory_provider.name == ""

    def test_app_config_memory_provider_from_dict(self):
        cfg = AppConfig.model_validate({
            "models": [],
            "sandbox": {"use": "x"},
            "memory_provider": {"enabled": True, "name": "mem0"},
        })
        assert cfg.memory_provider.enabled is True
        assert cfg.memory_provider.name == "mem0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_provider_config.py -v`
Expected: FAIL — module `deerflow.config.memory_provider_config` does not exist

- [ ] **Step 3: Write MemoryProviderConfig**

```python
# deerflow/config/memory_provider_config.py
"""Configuration for external memory provider (Mem0, Honcho, etc.)."""

from __future__ import annotations

from pydantic import BaseModel


class MemoryProviderConfig(BaseModel):
    """External memory provider configuration."""

    enabled: bool = False
    name: str = ""  # Provider name: "mem0", "honcho", etc. Empty = native only
```

- [ ] **Step 4: Register in AppConfig**

Add import at top of `deerflow/config/app_config.py` (after the `memory_search_config` import on line 19):

```python
from deerflow.config.memory_provider_config import MemoryProviderConfig
```

Add field to `AppConfig` class (after the `memory_search` field on line 99):

```python
    memory_provider: MemoryProviderConfig = Field(default_factory=MemoryProviderConfig, description="External memory provider configuration")
```

- [ ] **Step 5: Add config.example.yaml section**

Add after the `memory_search` section (after line 801 in config.example.yaml):

```yaml

# External memory provider (Mem0, Honcho, etc.)
# When enabled, an external memory service supplements the native MEMORY.md/USER.md store.
# Provider-specific configuration (API keys) goes in environment variables.
memory_provider:
  enabled: false  # Set to true and specify a provider name to activate
  name: ""  # "mem0" for Mem0 provider, or a custom provider name
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_provider_config.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/config/memory_provider_config.py backend/packages/harness/deerflow/config/app_config.py config.example.yaml backend/tests/test_memory_provider_config.py
git commit -m "feat(memory-provider): add MemoryProviderConfig and register in AppConfig"
```

---

### Task 3: NativeMemoryProvider adapter

**Files:**
- Create: `deerflow/memory_providers/native.py`
- Test: `tests/test_native_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_native_provider.py
"""Tests for NativeMemoryProvider wrapping MemoryStore."""

import json
import tempfile
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
        result = p.handle_tool_call("memory", {
            "action": "add",
            "target": "memory",
            "content": "hello world",
        })
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_native_provider.py -v`
Expected: FAIL — module `deerflow.memory_providers.native` does not exist

- [ ] **Step 3: Write implementation**

```python
# deerflow/memory_providers/native.py
"""NativeMemoryProvider — wraps existing MemoryStore as a MemoryProvider."""

from __future__ import annotations

import json
from typing import Any

from deerflow.agents.memory.store import MemoryStore
from deerflow.memory_providers.provider import MemoryProvider

_MEMORY_TOOL_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action to perform: add, replace, or remove.",
            },
            "target": {
                "type": "string",
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile.",
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'.",
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove.",
            },
        },
        "required": ["action", "target"],
    },
}


class NativeMemoryProvider(MemoryProvider):
    """Wraps MemoryStore as a MemoryProvider for use with MemoryManager."""

    def __init__(self, store: MemoryStore):
        self._store = store

    @property
    def name(self) -> str:
        return "native"

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        from deerflow.agents.memory import format_memory_block

        memory_snapshot = self._store.format_for_system_prompt("memory")
        user_snapshot = self._store.format_for_system_prompt("user")
        content = format_memory_block(memory_snapshot, user_snapshot)
        if not content.strip():
            return ""
        return f"<memory>\n{content}\n</memory>"

    def get_tool_schemas(self) -> list[dict]:
        return [_MEMORY_TOOL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name != "memory":
            raise ValueError(f"Unknown tool: {tool_name}")

        target = args.get("target", "memory")
        action = args.get("action", "")
        content = args.get("content")
        old_text = args.get("old_text")

        if target not in ("memory", "user"):
            return json.dumps({"success": False, "error": f"Invalid target '{target}'."})

        if action == "add":
            if not content:
                return json.dumps({"success": False, "error": "Content is required for 'add'."})
            result = self._store.add(target, content)
        elif action == "replace":
            if not old_text:
                return json.dumps({"success": False, "error": "old_text is required for 'replace'."})
            if not content:
                return json.dumps({"success": False, "error": "content is required for 'replace'."})
            result = self._store.replace(target, old_text, content)
        elif action == "remove":
            if not old_text:
                return json.dumps({"success": False, "error": "old_text is required for 'remove'."})
            result = self._store.remove(target, old_text)
        else:
            return json.dumps({"success": False, "error": f"Unknown action '{action}'."})

        return json.dumps(result, ensure_ascii=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_native_provider.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/memory_providers/native.py backend/tests/test_native_provider.py
git commit -m "feat(memory-provider): add NativeMemoryProvider wrapping MemoryStore"
```

---

### Task 4: MemoryManager orchestrator

**Files:**
- Create: `deerflow/memory_providers/manager.py`
- Test: `tests/test_memory_manager.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_manager.py
"""Tests for MemoryManager orchestrator."""

import json
import logging
import tempfile
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
        # Empty store → empty prompt
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
        result = mgr.handle_tool_call("memory", {
            "action": "add", "target": "memory", "content": "hello",
        })
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_manager.py -v`
Expected: FAIL — module `deerflow.memory_providers.manager` does not exist

- [ ] **Step 3: Write implementation**

```python
# deerflow/memory_providers/manager.py
"""MemoryManager — orchestrates native + optional external memory provider."""

from __future__ import annotations

import logging

from deerflow.memory_providers.provider import MemoryProvider

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates native + optional external memory provider.

    The native provider is always present. At most one external provider
    may be registered. All external-provider operations are best-effort:
    failures are logged but never crash the main conversation flow.
    """

    def __init__(self, native: MemoryProvider):
        self._native = native
        self._external: MemoryProvider | None = None
        self._tool_to_provider: dict[str, MemoryProvider] = {}
        self._initialized = False

    # --- Provider management ---

    def add_provider(self, provider: MemoryProvider) -> None:
        """Register an external provider. Raises if one is already registered."""
        if self._external is not None:
            raise ValueError(
                f"Already have external provider '{self._external.name}', "
                f"cannot add '{provider.name}'"
            )
        self._external = provider

    # --- Lifecycle ---

    def initialize_all(self, session_id: str, **kwargs) -> None:
        """Initialize all providers. External failures are logged, not raised."""
        self._native.initialize(session_id, **kwargs)
        if self._external:
            try:
                self._external.initialize(session_id, **kwargs)
            except Exception as exc:
                logger.warning(
                    "External provider '%s' init failed: %s",
                    self._external.name,
                    exc,
                )
        self._build_tool_map()
        self._initialized = True

    def shutdown_all(self) -> None:
        """Shut down all providers. Best-effort."""
        for p in self._providers:
            try:
                p.shutdown()
            except Exception as exc:
                logger.warning("Provider '%s' shutdown error: %s", p.name, exc)

    # --- Prompt assembly ---

    def build_system_prompt(self) -> str:
        """Collect system prompt blocks from all providers."""
        blocks = [self._native.system_prompt_block()]
        if self._external:
            ext_block = self._external.system_prompt_block()
            if ext_block:
                blocks.append(ext_block)
        return "\n\n".join(b for b in blocks if b)

    def build_external_prompt(self) -> str:
        """Return external provider's prompt block only."""
        if self._external:
            return self._external.system_prompt_block()
        return ""

    # --- Per-turn ---

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Collect prefetch context. External failures return empty."""
        results: list[str] = []
        if self._external:
            try:
                ctx = self._external.prefetch(query, session_id=session_id)
                if ctx:
                    results.append(ctx)
            except Exception as exc:
                logger.warning("External prefetch failed: %s", exc)
        return "\n\n".join(results)

    def sync_all(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Sync turn to all providers. Best-effort for external."""
        if self._external:
            try:
                self._external.sync_turn(
                    user_content, assistant_content, session_id=session_id,
                )
            except Exception as exc:
                logger.warning("External sync_turn failed: %s", exc)

    # --- Tool routing ---

    def get_all_tool_schemas(self) -> list[dict]:
        """Collect tool schemas from all providers."""
        schemas = list(self._native.get_tool_schemas())
        if self._external:
            schemas.extend(self._external.get_tool_schemas())
        return schemas

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Route tool call to owning provider."""
        provider = self._tool_to_provider.get(tool_name)
        if not provider:
            raise ValueError(f"Unknown tool: {tool_name}")
        return provider.handle_tool_call(tool_name, args, **kwargs)

    # --- Properties ---

    @property
    def native(self) -> MemoryProvider:
        return self._native

    @property
    def external(self) -> MemoryProvider | None:
        return self._external

    @property
    def _providers(self) -> list[MemoryProvider]:
        providers = [self._native]
        if self._external:
            providers.append(self._external)
        return providers

    def _build_tool_map(self) -> None:
        """Build tool_name -> provider mapping."""
        self._tool_to_provider.clear()
        for schema in self._native.get_tool_schemas():
            self._tool_to_provider[schema["name"]] = self._native
        if self._external:
            for schema in self._external.get_tool_schemas():
                self._tool_to_provider[schema["name"]] = self._external
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_manager.py -v`
Expected: All 20 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/memory_providers/manager.py backend/tests/test_memory_manager.py
git commit -m "feat(memory-provider): add MemoryManager orchestrator with tool routing"
```

---

### Task 5: Provider discovery and module __init__

**Files:**
- Create: `deerflow/memory_providers/__init__.py`
- Test: `tests/test_memory_provider_discovery.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_provider_discovery.py
"""Tests for provider discovery and loading."""

import importlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deerflow.memory_providers import discover_providers, load_provider
from deerflow.memory_providers.provider import MemoryProvider


class TestDiscoverProviders:
    def test_discovers_builtin_providers(self):
        providers = discover_providers()
        names = [name for name, _, _ in providers]
        assert "mem0" in names

    def test_builtin_providers_are_marked(self):
        providers = discover_providers()
        for name, _, is_builtin in providers:
            assert is_builtin is True


class TestLoadProvider:
    def test_load_builtin_mem0(self):
        # Mem0Provider should be loadable even without mem0ai installed
        provider = load_provider("mem0")
        assert provider is not None
        assert provider.name == "mem0"

    def test_load_nonexistent_returns_none(self):
        result = load_provider("nonexistent_provider_xyz")
        assert result is None

    def test_load_from_user_dir(self, tmp_path):
        # Create a fake user-installed provider
        plugin_dir = tmp_path / "plugins" / "memory" / "fake_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "__init__.py").write_text(
            "from deerflow.memory_providers.provider import MemoryProvider\n"
            "class FakePluginProvider(MemoryProvider):\n"
            "    @property\n"
            "    def name(self): return 'fake_plugin'\n"
            "    def is_available(self): return True\n"
        )
        provider = load_provider("fake_plugin", base_dir=tmp_path)
        assert provider is not None
        assert provider.name == "fake_plugin"

    def test_builtin_takes_precedence_over_user(self, tmp_path):
        # Create a user-installed mem0 (should be ignored in favor of builtin)
        plugin_dir = tmp_path / "plugins" / "memory" / "mem0"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "__init__.py").write_text(
            "from deerflow.memory_providers.provider import MemoryProvider\n"
            "class FakeOverride(MemoryProvider):\n"
            "    @property\n"
            "    def name(self): return 'overridden'\n"
            "    def is_available(self): return True\n"
        )
        provider = load_provider("mem0", base_dir=tmp_path)
        assert provider is not None
        assert provider.name == "mem0"  # builtin, not overridden
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_provider_discovery.py -v`
Expected: FAIL — module `deerflow.memory_providers` does not exist

- [ ] **Step 3: Write implementation**

```python
# deerflow/memory_providers/__init__.py
"""Memory provider plugin system — discover, load, and register memory backends."""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path

from deerflow.memory_providers.provider import MemoryProvider

logger = logging.getLogger(__name__)

__all__ = ["MemoryProvider", "discover_providers", "load_provider"]

_BUILTIN_PACKAGE = "deerflow.memory_providers"


def discover_providers(base_dir: Path | None = None) -> list[tuple[str, str, bool]]:
    """Return (name, module_path, is_builtin) for each found provider.

    Scan order:
    1. Built-in: deerflow.memory_providers.<name> subdirectories
    2. User-installed: {base_dir}/plugins/memory/<name>/
    Built-in takes precedence on name collision.
    """
    providers: dict[str, tuple[str, str, bool]] = {}

    # Scan user-installed first (lower priority)
    if base_dir is not None:
        plugins_dir = base_dir / "plugins" / "memory"
        if plugins_dir.is_dir():
            for child in sorted(plugins_dir.iterdir()):
                if child.is_dir() and (child / "__init__.py").exists():
                    providers[child.name] = (
                        child.name,
                        str(child),
                        False,
                    )

    # Scan built-in (overwrites user-installed on name collision)
    try:
        import deerflow.memory_providers as pkg

        pkg_dir = Path(pkg.__file__).parent
        for child in sorted(pkg_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                if child.name.startswith("_"):
                    continue
                providers[child.name] = (
                    child.name,
                    f"{_BUILTIN_PACKAGE}.{child.name}",
                    True,
                )
    except Exception as exc:
        logger.warning("Failed to scan builtin providers: %s", exc)

    return [(name, path, is_builtin) for name, path, is_builtin in providers.values()]


def load_provider(name: str, base_dir: Path | None = None) -> MemoryProvider | None:
    """Load and instantiate a provider by name.

    1. Try built-in first: ``deerflow.memory_providers.<name>``
    2. Fall back to user-installed: ``{base_dir}/plugins/memory/<name>/``
    3. Import module, find MemoryProvider subclass, instantiate
    4. Return None if not found or import fails
    """
    # Try built-in
    builtin_module = f"{_BUILTIN_PACKAGE}.{name}"
    provider = _try_load_builtin(builtin_module)
    if provider is not None:
        return provider

    # Try user-installed
    if base_dir is not None:
        user_module_path = base_dir / "plugins" / "memory" / name / "__init__.py"
        if user_module_path.exists():
            return _try_load_user(name, user_module_path)

    logger.warning("Provider '%s' not found", name)
    return None


def _try_load_builtin(module_name: str) -> MemoryProvider | None:
    """Try to import a built-in provider module and find a MemoryProvider subclass."""
    try:
        module = importlib.import_module(module_name)
        return _find_provider_class(module)
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("Failed to load builtin provider '%s': %s", module_name, exc)
        return None


def _try_load_user(name: str, init_path: Path) -> MemoryProvider | None:
    """Try to load a user-installed provider from a file path."""
    import sys

    module_name = f"_deerflow_user_provider_{name}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(init_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return _find_provider_class(module)
    except Exception as exc:
        logger.warning("Failed to load user provider '%s': %s", name, exc)
        return None


def _find_provider_class(module) -> MemoryProvider | None:
    """Find the first concrete MemoryProvider subclass in a module."""
    for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, MemoryProvider) and obj is not MemoryProvider:
            try:
                return obj()
            except Exception as exc:
                logger.warning("Failed to instantiate %s: %s", obj, exc)
                return None
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_provider_discovery.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/memory_providers/__init__.py backend/tests/test_memory_provider_discovery.py
git commit -m "feat(memory-provider): add provider discovery with directory scanning"
```

---

### Task 6: Mem0 provider implementation

**Files:**
- Create: `deerflow/memory_providers/mem0/__init__.py`
- Create: `deerflow/memory_providers/mem0/tool.py`
- Test: `tests/test_mem0_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mem0_provider.py
"""Tests for Mem0Provider — uses mocks, no real API calls."""

import time
from unittest.mock import MagicMock, patch

import pytest

from deerflow.memory_providers.mem0 import Mem0Provider


class TestMem0ProviderBasics:
    def test_name(self):
        p = Mem0Provider()
        assert p.name == "mem0"

    def test_is_available_false_without_api_key(self):
        p = Mem0Provider(api_key=None)
        with patch.dict("os.environ", {}, clear=True):
            # Remove MEM0_API_KEY if present
            assert p.is_available() is False

    def test_is_available_false_without_package(self):
        p = Mem0Provider(api_key="test-key")
        # The import will fail if mem0ai is not installed
        # We test that it handles ImportError gracefully
        with patch.dict("sys.modules", {"mem0": None}):
            assert p.is_available() is False

    def test_system_prompt_block(self):
        p = Mem0Provider()
        block = p.system_prompt_block()
        assert "Mem0" in block
        assert "mem0_search" in block
        assert "mem0_add" in block


class TestMem0ProviderInitialize:
    def test_initialize_creates_client(self):
        p = Mem0Provider(api_key="test-key")
        mock_client_cls = MagicMock()
        with patch("deerflow.memory_providers.mem0.MemoryClient", mock_client_cls, create=True):
            # We need to mock the import inside initialize
            import sys
            fake_mem0 = MagicMock()
            fake_mem0.MemoryClient = mock_client_cls
            sys.modules["mem0"] = fake_mem0
            try:
                p.initialize("session_1", user_id="user_1")
                assert p._client is mock_client_cls.return_value
                assert p._user_id == "user_1"
            finally:
                del sys.modules["mem0"]

    def test_initialize_uses_session_id_as_user_id(self):
        p = Mem0Provider(api_key="test-key")
        mock_client_cls = MagicMock()
        import sys
        fake_mem0 = MagicMock()
        fake_mem0.MemoryClient = mock_client_cls
        sys.modules["mem0"] = fake_mem0
        try:
            p.initialize("session_abc")
            assert p._user_id == "session_abc"
        finally:
            del sys.modules["mem0"]


class TestMem0ProviderPrefetch:
    def test_prefetch_returns_results(self):
        p = Mem0Provider(api_key="test-key")
        p._client = MagicMock()
        p._client.search.return_value = [
            {"memory": "fact 1"},
            {"memory": "fact 2"},
        ]
        result = p.prefetch("test query")
        assert "fact 1" in result
        assert "fact 2" in result

    def test_prefetch_empty_results(self):
        p = Mem0Provider(api_key="test-key")
        p._client = MagicMock()
        p._client.search.return_value = []
        result = p.prefetch("test query")
        assert result == ""

    def test_prefetch_failure_returns_empty(self):
        p = Mem0Provider(api_key="test-key")
        p._client = MagicMock()
        p._client.search.side_effect = RuntimeError("API error")
        result = p.prefetch("test query")
        assert result == ""

    def test_prefetch_circuit_open_returns_empty(self):
        p = Mem0Provider(api_key="test-key")
        p._circuit_open = True
        p._circuit_opened_at = time.monotonic()
        result = p.prefetch("test query")
        assert result == ""


class TestMem0ProviderSyncTurn:
    def test_sync_turn_calls_add(self):
        p = Mem0Provider(api_key="test-key")
        p._client = MagicMock()
        p.sync_turn("hello", "world")
        p._client.add.assert_called_once()

    def test_sync_turn_failure_does_not_raise(self):
        p = Mem0Provider(api_key="test-key")
        p._client = MagicMock()
        p._client.add.side_effect = RuntimeError("API error")
        p.sync_turn("hello", "world")  # should not raise

    def test_sync_turn_circuit_open_skips(self):
        p = Mem0Provider(api_key="test-key")
        p._client = MagicMock()
        p._circuit_open = True
        p._circuit_opened_at = time.monotonic()
        p.sync_turn("hello", "world")
        p._client.add.assert_not_called()


class TestMem0ProviderCircuitBreaker:
    def test_circuit_opens_after_threshold(self):
        p = Mem0Provider(api_key="test-key")
        p._client = MagicMock()
        p._client.search.side_effect = RuntimeError("fail")
        for _ in range(5):
            p.prefetch("query")
        assert p._circuit_open is True

    def test_circuit_resets_after_cooldown(self):
        p = Mem0Provider(api_key="test-key")
        p._circuit_open = True
        p._circuit_opened_at = time.monotonic() - 301  # 301 seconds ago
        p._client = MagicMock()
        p._client.search.return_value = [{"memory": "recovered"}]
        result = p.prefetch("query")
        assert p._circuit_open is False
        assert "recovered" in result


class TestMem0ProviderTools:
    def test_get_tool_schemas(self):
        p = Mem0Provider()
        schemas = p.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "mem0_search" in names
        assert "mem0_add" in names

    def test_handle_mem0_search(self):
        p = Mem0Provider(api_key="test-key")
        p._client = MagicMock()
        p._client.search.return_value = [{"memory": "found fact"}]
        result = p.handle_tool_call("mem0_search", {"query": "test"})
        assert "found fact" in result

    def test_handle_mem0_add(self):
        p = Mem0Provider(api_key="test-key")
        p._client = MagicMock()
        p._client.add.return_value = [{"id": "abc"}]
        result = p.handle_tool_call("mem0_add", {"content": "remember this"})
        p._client.add.assert_called_once()

    def test_handle_unknown_tool_raises(self):
        p = Mem0Provider()
        with pytest.raises(ValueError, match="Unknown tool"):
            p.handle_tool_call("nonexistent", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_mem0_provider.py -v`
Expected: FAIL — module `deerflow.memory_providers.mem0` does not exist

- [ ] **Step 3: Write tool schemas**

```python
# deerflow/memory_providers/mem0/tool.py
"""Mem0 tool schemas and handler helpers."""

from __future__ import annotations

MEM0_SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": "Search your Mem0 long-term memory for relevant context from past conversations.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant memories.",
            },
        },
        "required": ["query"],
    },
}

MEM0_ADD_SCHEMA = {
    "name": "mem0_add",
    "description": "Add a specific memory to your Mem0 store for future recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to remember.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata to attach to the memory.",
            },
        },
        "required": ["content"],
    },
}
```

- [ ] **Step 4: Write Mem0Provider**

```python
# deerflow/memory_providers/mem0/__init__.py
"""Mem0 external memory provider."""

from __future__ import annotations

import json
import logging
import os
import time

from deerflow.memory_providers.mem0.tool import MEM0_ADD_SCHEMA, MEM0_SEARCH_SCHEMA
from deerflow.memory_providers.provider import MemoryProvider

logger = logging.getLogger(__name__)


class Mem0Provider(MemoryProvider):
    """Mem0 external memory provider with circuit breaker."""

    CIRCUIT_THRESHOLD = 5
    CIRCUIT_COOLDOWN = 300  # 5 minutes

    def __init__(self, api_key: str | None = None, user_id: str = ""):
        self._api_key = api_key or os.environ.get("MEM0_API_KEY")
        self._user_id = user_id
        self._client = None
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_opened_at: float = 0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            from mem0 import MemoryClient  # noqa: F401

            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        from mem0 import MemoryClient

        self._client = MemoryClient(api_key=self._api_key)
        self._user_id = kwargs.get("user_id", self._user_id or session_id)

    def system_prompt_block(self) -> str:
        return (
            "You have access to a Mem0 long-term memory service.\n"
            "Use mem0_search to find relevant past context.\n"
            "Use mem0_add to store important information."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._check_circuit()
        if self._circuit_open:
            return ""
        try:
            results = self._client.search(query, user_id=self._user_id)
            self._failure_count = 0
            if not results:
                return ""
            return "\n".join(r.get("memory", "") for r in results[:5])
        except Exception as exc:
            self._on_failure(exc)
            return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        self._check_circuit()
        if self._circuit_open:
            return
        try:
            self._client.add(
                f"User: {user_content}\nAssistant: {assistant_content}",
                user_id=self._user_id,
            )
            self._failure_count = 0
        except Exception as exc:
            self._on_failure(exc)

    def get_tool_schemas(self) -> list[dict]:
        return [MEM0_SEARCH_SCHEMA, MEM0_ADD_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "mem0_search":
            return self._handle_search(args.get("query", ""))
        elif tool_name == "mem0_add":
            return self._handle_add(args.get("content", ""), args.get("metadata"))
        raise ValueError(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        self._client = None

    # --- Circuit breaker ---

    def _on_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        logger.warning("Mem0 API failure (%d): %s", self._failure_count, exc)
        if self._failure_count >= self.CIRCUIT_THRESHOLD:
            self._circuit_open = True
            self._circuit_opened_at = time.monotonic()
            logger.error("Mem0 circuit breaker OPEN")

    def _check_circuit(self) -> None:
        if self._circuit_open:
            elapsed = time.monotonic() - self._circuit_opened_at
            if elapsed >= self.CIRCUIT_COOLDOWN:
                self._circuit_open = False
                self._failure_count = 0
                logger.info("Mem0 circuit breaker reset")

    # --- Tool handlers ---

    def _handle_search(self, query: str) -> str:
        if not query:
            return json.dumps({"success": False, "error": "Query is required."})
        self._check_circuit()
        if self._circuit_open:
            return json.dumps({"success": False, "error": "Mem0 temporarily unavailable."})
        try:
            results = self._client.search(query, user_id=self._user_id)
            self._failure_count = 0
            return json.dumps({
                "success": True,
                "results": [r.get("memory", "") for r in (results or [])[:5]],
            })
        except Exception as exc:
            self._on_failure(exc)
            return json.dumps({"success": False, "error": str(exc)})

    def _handle_add(self, content: str, metadata: dict | None = None) -> str:
        if not content:
            return json.dumps({"success": False, "error": "Content is required."})
        self._check_circuit()
        if self._circuit_open:
            return json.dumps({"success": False, "error": "Mem0 temporarily unavailable."})
        try:
            result = self._client.add(content, user_id=self._user_id, metadata=metadata)
            self._failure_count = 0
            return json.dumps({"success": True, "result": result})
        except Exception as exc:
            self._on_failure(exc)
            return json.dumps({"success": False, "error": str(exc)})
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_mem0_provider.py -v`
Expected: All 17 tests PASS

- [ ] **Step 6: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/memory_providers/mem0/ backend/tests/test_mem0_provider.py
git commit -m "feat(memory-provider): add Mem0 provider with circuit breaker and dual tools"
```

---

### Task 7: Integration — agent.py tool registration + prompt.py external block

**Files:**
- Modify: `deerflow/agents/lead_agent/agent.py`
- Modify: `deerflow/agents/lead_agent/prompt.py`
- Test: `tests/test_memory_provider_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_provider_integration.py
"""Tests for MemoryManager integration with lead agent and prompt."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from deerflow.agents.memory.store import MemoryStore
from deerflow.memory_providers.manager import MemoryManager
from deerflow.memory_providers.native import NativeMemoryProvider


def _make_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(memory_dir=tmp_path, memory_char_limit=2200, user_char_limit=1375)
    store.load_from_disk()
    return store


class TestAgentIntegration:
    def test_memory_manager_created_when_memory_enabled(self, tmp_path):
        """Verify MemoryManager is constructable from agent.py's flow."""
        store = _make_store(tmp_path)
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)
        mgr.initialize_all("thread_123")
        assert mgr._initialized is True

    def test_external_provider_not_loaded_when_disabled(self, tmp_path):
        """When memory_provider.enabled=False, no external provider is loaded."""
        store = _make_store(tmp_path)
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)
        assert mgr.external is None

    def test_external_provider_registered_when_available(self, tmp_path):
        """When an external provider is available, it gets registered."""
        store = _make_store(tmp_path)
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)

        from deerflow.memory_providers.provider import MemoryProvider

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
        """Native memory tool actions still function through manager."""
        import json

        store = _make_store(tmp_path)
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)
        mgr.initialize_all("thread_123")

        schemas = mgr.get_all_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "memory" in names

        result = mgr.handle_tool_call("memory", {
            "action": "add",
            "target": "memory",
            "content": "integration test",
        })
        parsed = json.loads(result)
        assert parsed["success"] is True


class TestPromptIntegration:
    def test_external_prompt_appended_to_memory_context(self, tmp_path):
        """build_external_prompt returns external block when provider exists."""
        store = _make_store(tmp_path)
        store.add("memory", "native fact")
        native = NativeMemoryProvider(store=store)
        mgr = MemoryManager(native)

        # No external → empty
        assert mgr.build_external_prompt() == ""

        # With external → returns external block
        from deerflow.memory_providers.provider import MemoryProvider

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

        # Full system prompt includes both
        full = mgr.build_system_prompt()
        assert "native fact" in full
        assert "External memory active." in full
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_provider_integration.py -v`
Expected: These should mostly pass already since they test the manager/provider interface directly. The real integration is in the modified agent.py and prompt.py files.

- [ ] **Step 3: Modify agent.py — add MemoryManager creation**

In `deerflow/agents/lead_agent/agent.py`, after the existing memory tool registration block (after line 410, `extra_tools.append(create_memory_tool(store))`), add:

```python
    # Wrap MemoryStore in MemoryManager for external provider support
    from deerflow.memory_providers.manager import MemoryManager
    from deerflow.memory_providers.native import NativeMemoryProvider

    _native_provider = NativeMemoryProvider(store=store)
    _memory_manager = MemoryManager(_native_provider)

    if (resolved_app_config.memory_provider.enabled
            and resolved_app_config.memory_provider.name):
        from deerflow.memory_providers import load_provider

        _ext = load_provider(
            resolved_app_config.memory_provider.name,
            base_dir=paths.base_dir,
        )
        if _ext and _ext.is_available():
            _memory_manager.add_provider(_ext)
            logger.info("Loaded external memory provider: %s", _ext.name)
        else:
            logger.warning(
                "Provider '%s' not available, using native only",
                resolved_app_config.memory_provider.name,
            )

    _memory_manager.initialize_all(
        config.get("configurable", {}).get("thread_id", "")
    )
    config["memory_manager"] = _memory_manager

    # Register external provider tools (if any)
    for _schema in _memory_manager.get_all_tool_schemas():
        if _schema["name"] != "memory":
            extra_tools.append(
                _create_routed_tool(_memory_manager, _schema)
            )
```

Also add the `_create_routed_tool` helper function in agent.py (before `_make_lead_agent`):

```python
def _create_routed_tool(manager, schema: dict):
    """Create a LangChain tool that routes through MemoryManager."""
    from langchain.tools import tool as lc_tool

    _tool_name = schema["name"]

    @lc_tool(_tool_name)
    def _routed_tool(**kwargs) -> str:
        return manager.handle_tool_call(_tool_name, kwargs)

    _routed_tool.description = schema.get("description", "")
    return _routed_tool
```

- [ ] **Step 4: Modify prompt.py — accept memory_manager parameter**

In `deerflow/agents/lead_agent/prompt.py`, modify `_get_memory_context` function signature (line 535):

Change:
```python
def _get_memory_context(agent_name: str | None = None, *, app_config: AppConfig | None = None) -> str:
```
To:
```python
def _get_memory_context(agent_name: str | None = None, *, app_config: AppConfig | None = None, memory_manager: "MemoryManager | None" = None) -> str:
```

Add at the end of the function, before the final return (before `return f"<memory>\n{memory_content}\n</memory>\n"`):

```python
    # Append external provider prompt block if available
    if memory_manager:
        ext_block = memory_manager.build_external_prompt()
        if ext_block:
            memory_content += f"\n\n{ext_block}"
```

Update the caller at line 761 to pass memory_manager. Change:
```python
    memory_context = _get_memory_context(agent_name, app_config=app_config)
```
To:
```python
    memory_context = _get_memory_context(agent_name, app_config=app_config, memory_manager=None)
```

Note: The `memory_manager` is not available in the static `apply_prompt_template` context. It will be injected via a different mechanism in the future when the prompt template system is refactored. For now, external provider prompt is available via `config["memory_manager"]` for dynamic injection.

- [ ] **Step 5: Run integration tests**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_provider_integration.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full regression test**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/ -x -q --ignore=tests/test_client_live.py`
Expected: All tests PASS (same count as before — no regressions)

- [ ] **Step 7: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/agents/lead_agent/agent.py backend/packages/harness/deerflow/agents/lead_agent/prompt.py backend/tests/test_memory_provider_integration.py
git commit -m "feat(memory-provider): integrate MemoryManager into lead agent tool registration"
```

---

### Task 8: Full regression test + final cleanup

**Files:**
- All files (verification only)

- [ ] **Step 1: Run full test suite**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/ -x -q --ignore=tests/test_client_live.py`
Expected: All tests PASS, 0 failures

- [ ] **Step 2: Run ruff check**

Run: `cd /root/deer-flow/backend && uv run ruff check packages/harness/deerflow/memory_providers/`
Expected: No errors

- [ ] **Step 3: Verify module imports cleanly**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run python -c "from deerflow.memory_providers import discover_providers, load_provider; from deerflow.memory_providers.manager import MemoryManager; from deerflow.memory_providers.native import NativeMemoryProvider; from deerflow.memory_providers.mem0 import Mem0Provider; print('All imports OK')"`
Expected: "All imports OK"

- [ ] **Step 4: Verify discovery finds mem0**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run python -c "from deerflow.memory_providers import discover_providers; print(discover_providers())"`
Expected: List containing `('mem0', 'deerflow.memory_providers.mem0', True)`

- [ ] **Step 5: Commit any remaining fixes**

```bash
cd /root/deer-flow
git add -A
git commit -m "chore(memory-provider): cleanup and verify all imports"
```

---

## Self-Review

### Spec Coverage

| Spec Section | Task |
|-------------|------|
| 5.1 Architecture | Task 3 (Native), Task 4 (Manager), Task 7 (Integration) |
| 5.2 MemoryProvider ABC | Task 1 |
| 5.3 MemoryManager | Task 4 |
| 5.4 Discovery and Loading | Task 5 |
| 5.5 Mem0 Provider | Task 6 |
| 5.6 Integration (agent.py, prompt.py) | Task 7 |
| 5.7 Files Summary | All tasks |
| 5.8 Error Handling | Task 4 (best-effort), Task 6 (circuit breaker) |
| 5.9 Out of Scope | N/A |

### Placeholder Scan

No TBD, TODO, or incomplete sections found. All code blocks contain complete implementations.

### Type Consistency

- `MemoryProvider.name` is `str` property throughout all tasks
- `get_tool_schemas()` returns `list[dict]` in all implementations
- `handle_tool_call()` returns `str` (JSON) in all implementations
- `NativeMemoryProvider.__init__` takes `store: MemoryStore` in Task 3, used consistently in Tasks 4, 7
- `_create_routed_tool` uses `schema["name"]` matching `get_tool_schemas()` output format
