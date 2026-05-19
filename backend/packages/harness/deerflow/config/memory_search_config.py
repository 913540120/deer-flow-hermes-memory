"""Configuration for session search (FTS5 full-text search over conversation history)."""

from __future__ import annotations

from pydantic import BaseModel


class MemorySearchConfig(BaseModel):
    """Session search configuration."""

    enabled: bool = False
    db_path: str = ".deer-flow/data/search.db"
    max_results: int = 3
    max_content_chars: int = 100_000
