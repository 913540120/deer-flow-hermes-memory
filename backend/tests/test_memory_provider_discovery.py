"""Tests for provider discovery and loading."""

from deerflow.memory_providers import discover_providers, load_provider


class TestDiscoverProviders:
    def test_discovers_builtin_providers(self):
        providers = discover_providers()
        names = [name for name, _, _ in providers]
        assert "mem0" in names

    def test_builtin_providers_are_marked(self):
        providers = discover_providers()
        for name, _, is_builtin in providers:
            assert is_builtin is True


class TestLoadProvider:
    def test_load_builtin_mem0(self):
        provider = load_provider("mem0")
        assert provider is not None
        assert provider.name == "mem0"

    def test_load_nonexistent_returns_none(self):
        result = load_provider("nonexistent_provider_xyz")
        assert result is None

    def test_load_from_user_dir(self, tmp_path):
        plugin_dir = tmp_path / "plugins" / "memory" / "fake_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "__init__.py").write_text(
            "from deerflow.memory_providers.provider import MemoryProvider\nclass FakePluginProvider(MemoryProvider):\n    @property\n    def name(self): return 'fake_plugin'\n    def is_available(self): return True\n"
        )
        provider = load_provider("fake_plugin", base_dir=tmp_path)
        assert provider is not None
        assert provider.name == "fake_plugin"

    def test_builtin_takes_precedence_over_user(self, tmp_path):
        plugin_dir = tmp_path / "plugins" / "memory" / "mem0"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "__init__.py").write_text(
            "from deerflow.memory_providers.provider import MemoryProvider\nclass FakeOverride(MemoryProvider):\n    @property\n    def name(self): return 'overridden'\n    def is_available(self): return True\n"
        )
        provider = load_provider("mem0", base_dir=tmp_path)
        assert provider is not None
        assert provider.name == "mem0"  # builtin, not overridden
