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
