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
