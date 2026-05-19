"""Memory provider plugin system — discover, load, and register memory backends."""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
from pathlib import Path

from deerflow.memory_providers.provider import MemoryProvider

logger = logging.getLogger(__name__)

__all__ = ["MemoryProvider", "discover_providers", "load_provider"]

_BUILTIN_PACKAGE = "deerflow.memory_providers"


def discover_providers(base_dir: Path | None = None) -> list[tuple[str, str, bool]]:
    """Return (name, module_path, is_builtin) for each found provider.

    Scan order:
    1. Built-in: deerflow.memory_providers.<name> subdirectories
    2. User-installed: {base_dir}/plugins/memory/<name>/
    Built-in takes precedence on name collision.
    """
    providers: dict[str, tuple[str, str, bool]] = {}

    # Scan user-installed first (lower priority)
    if base_dir is not None:
        plugins_dir = base_dir / "plugins" / "memory"
        if plugins_dir.is_dir():
            for child in sorted(plugins_dir.iterdir()):
                if child.is_dir() and (child / "__init__.py").exists():
                    providers[child.name] = (
                        child.name,
                        str(child),
                        False,
                    )

    # Scan built-in (overwrites user-installed on name collision)
    try:
        import deerflow.memory_providers as pkg

        pkg_dir = Path(pkg.__file__).parent
        for child in sorted(pkg_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                if child.name.startswith("_"):
                    continue
                providers[child.name] = (
                    child.name,
                    f"{_BUILTIN_PACKAGE}.{child.name}",
                    True,
                )
    except Exception as exc:
        logger.warning("Failed to scan builtin providers: %s", exc)

    return [(name, path, is_builtin) for name, path, is_builtin in providers.values()]


def load_provider(name: str, base_dir: Path | None = None) -> MemoryProvider | None:
    """Load and instantiate a provider by name.

    1. Try built-in first: ``deerflow.memory_providers.<name>``
    2. Fall back to user-installed: ``{base_dir}/plugins/memory/<name>/``
    3. Return None if not found or import fails
    """
    builtin_module = f"{_BUILTIN_PACKAGE}.{name}"
    provider = _try_load_builtin(builtin_module)
    if provider is not None:
        return provider

    if base_dir is not None:
        user_module_path = base_dir / "plugins" / "memory" / name / "__init__.py"
        if user_module_path.exists():
            return _try_load_user(name, user_module_path)

    logger.warning("Provider '%s' not found", name)
    return None


def _try_load_builtin(module_name: str) -> MemoryProvider | None:
    try:
        module = importlib.import_module(module_name)
        return _find_provider_class(module)
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("Failed to load builtin provider '%s': %s", module_name, exc)
        return None


def _try_load_user(name: str, init_path: Path) -> MemoryProvider | None:
    module_name = f"_deerflow_user_provider_{name}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(init_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return _find_provider_class(module)
    except Exception as exc:
        logger.warning("Failed to load user provider '%s': %s", name, exc)
        return None


def _find_provider_class(module) -> MemoryProvider | None:
    for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, MemoryProvider) and obj is not MemoryProvider:
            try:
                return obj()
            except Exception as exc:
                logger.warning("Failed to instantiate %s: %s", obj, exc)
                return None
    return None
