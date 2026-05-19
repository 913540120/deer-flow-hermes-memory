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
            raise ValueError(f"Already have external provider '{self._external.name}', cannot add '{provider.name}'")
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
                    user_content,
                    assistant_content,
                    session_id=session_id,
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
