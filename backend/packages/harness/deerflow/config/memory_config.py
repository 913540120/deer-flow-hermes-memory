"""Memory system configuration — Hermes-style."""

from __future__ import annotations

from pydantic import BaseModel

_global_memory_config: MemoryConfig | None = None


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    enabled: bool = True
    injection_enabled: bool = True
    storage_path: str = ".deer-flow"
    memory_char_limit: int = 2200
    user_char_limit: int = 1375
    nudge_interval: int = 10


def get_memory_config() -> MemoryConfig:
    """Get the global memory configuration, initializing from app config if needed."""
    global _global_memory_config
    if _global_memory_config is None:
        from deerflow.config.app_config import get_app_config

        app_config = get_app_config()
        _global_memory_config = app_config.memory
    return _global_memory_config


def set_memory_config(config: MemoryConfig) -> None:
    """Set the global memory configuration (for testing)."""
    global _global_memory_config
    _global_memory_config = config


def load_memory_config_from_dict(data: dict) -> MemoryConfig:
    """Load memory config from a dictionary (e.g., parsed YAML).

    Also updates the global singleton so subsequent get_memory_config() calls
    return the newly loaded config.
    """
    global _global_memory_config
    _global_memory_config = MemoryConfig(**{k: v for k, v in data.items() if k in MemoryConfig.model_fields})
    return _global_memory_config
