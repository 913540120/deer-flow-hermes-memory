"""Sync message indexer for RunJournal integration."""

import logging
import time
from typing import Any

from deerflow.memory_search.storage import SearchStorage

logger = logging.getLogger(__name__)


class SearchIndexer:
    """Writes messages from RunJournal events into the search database."""

    def __init__(self, storage: SearchStorage) -> None:
        self._storage = storage

    def write_message(self, event: dict[str, Any]) -> None:
        try:
            self._do_write(event)
        except Exception:
            logger.warning(
                "Failed to index message for search: event_type=%s",
                event.get("event_type"),
                exc_info=True,
            )

    def _do_write(self, event: dict[str, Any]) -> None:
        category = event.get("category")
        if category != "message":
            return

        content = event.get("content")
        if not isinstance(content, dict):
            return

        event_type = event.get("event_type", "")
        thread_id = event.get("thread_id")
        if not thread_id:
            return

        role = self._extract_role(event_type, content)
        if not role:
            return

        text = self._extract_text(content)
        if not text or not text.strip():
            return

        user_id = event.get("metadata", {}).get("user_id")
        model = event.get("metadata", {}).get("model")
        tool_name = content.get("name") if role == "tool" else None
        timestamp = time.time()

        self._storage.upsert_session(
            session_id=thread_id,
            source="api",
            user_id=user_id,
            model=model,
            started_at=timestamp,
        )

        self._storage.insert_message(
            session_id=thread_id,
            role=role,
            content=text[:50000],
            tool_name=tool_name,
            timestamp=timestamp,
        )

    @staticmethod
    def _extract_role(event_type: str, content: dict) -> str | None:
        msg_type = content.get("type")
        if msg_type:
            return msg_type

        if "human" in event_type or "human.input" in event_type:
            return "user"
        if "ai.response" in event_type:
            return "assistant"
        if "tool.result" in event_type:
            return "tool"
        return None

    @staticmethod
    def _extract_text(content: dict) -> str | None:
        text = content.get("content")
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            parts = []
            for item in text:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(parts) if parts else None
        return None
