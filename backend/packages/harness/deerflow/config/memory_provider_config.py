"""Configuration for external memory provider (Mem0, Honcho, etc.)."""

from __future__ import annotations

from pydantic import BaseModel


class MemoryProviderConfig(BaseModel):
    """External memory provider configuration."""

    enabled: bool = False
    name: str = ""  # Provider name: "mem0", "honcho", etc. Empty = native only
