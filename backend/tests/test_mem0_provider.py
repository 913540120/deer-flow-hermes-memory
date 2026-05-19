"""Tests for Mem0Provider — uses mocks, no real API calls."""

import sys
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
            assert p.is_available() is False

    def test_is_available_false_without_package(self):
        p = Mem0Provider(api_key="test-key")
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
        p.handle_tool_call("mem0_add", {"content": "remember this"})
        p._client.add.assert_called_once()

    def test_handle_unknown_tool_raises(self):
        p = Mem0Provider()
        with pytest.raises(ValueError, match="Unknown tool"):
            p.handle_tool_call("nonexistent", {})
