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
        cfg = AppConfig.model_validate(
            {
                "models": [],
                "sandbox": {"use": "x"},
            }
        )
        assert hasattr(cfg, "memory_provider")
        assert cfg.memory_provider.enabled is False
        assert cfg.memory_provider.name == ""

    def test_app_config_memory_provider_from_dict(self):
        cfg = AppConfig.model_validate(
            {
                "models": [],
                "sandbox": {"use": "x"},
                "memory_provider": {"enabled": True, "name": "mem0"},
            }
        )
        assert cfg.memory_provider.enabled is True
        assert cfg.memory_provider.name == "mem0"
