"""Mem0 external memory provider with circuit breaker."""

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
        return "You have access to a Mem0 long-term memory service.\nUse mem0_search to find relevant past context.\nUse mem0_add to store important information."

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
            return json.dumps(
                {
                    "success": True,
                    "results": [r.get("memory", "") for r in (results or [])[:5]],
                }
            )
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
