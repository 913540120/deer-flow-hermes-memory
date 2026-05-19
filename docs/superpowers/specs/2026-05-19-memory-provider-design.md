# Phase 5 Design: External Memory Provider Plugin Architecture

Pluggable memory backend system — Phase 5 of the DeerFlow memory architecture alignment.

## Overview

Phase 5 adds a `MemoryProvider(ABC)` plugin interface and `MemoryManager` orchestrator,
enabling DeerFlow to use external memory services (Mem0, Honcho, etc.) alongside the
native MEMORY.md/USER.md store. The architecture wraps the existing `MemoryStore` as
`NativeMemoryProvider` and supports loading at most one external provider at a time.

---

## 5.1 Architecture

```
┌─────────────────────────────────────────────────┐
│                  Lead Agent                      │
│  ┌───────────────────────────────────────────┐   │
│  │            MemoryManager                  │   │
│  │                                           │   │
│  │  ┌─────────────┐  ┌──────────────────┐   │   │
│  │  │   Native    │  │  External (opt)   │   │   │
│  │  │  Provider   │  │  e.g. Mem0        │   │   │
│  │  │ (wrap       │  │                   │   │   │
│  │  │  MemoryStore)│  │                   │   │   │
│  │  └──────┬──────┘  └────────┬──────────┘   │   │
│  │         │                  │               │   │
│  │    ┌────▼──────────────────▼────┐          │   │
│  │    │  Tool routing / lifecycle  │          │   │
│  │    └───────────────────────────┘          │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
│  Tools: memory (native) + provider tools (opt)    │
│  Prompt: native snapshot + provider.system_block  │
└─────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Core subset of Hermes API**: 9 methods total — 2 abstract (`name`, `is_available`)
   + 7 with default implementations (`initialize`, `shutdown`, `system_prompt_block`,
   `prefetch`, `sync_turn`, `get_tool_schemas`, `handle_tool_call`). Skips Hermes-specific
   multi-session hooks
   (`on_delegation`, `on_session_switch`, `on_pre_compress`, etc.) that DeerFlow
   doesn't need.
2. **Single external provider limit**: Native provider always present; at most one
   external provider active. Simplifies tool routing and error handling.
3. **Wrapping integration**: `NativeMemoryProvider` wraps existing `MemoryStore`
   without modifying it. All existing memory functionality (tool, scrubber, nudge,
   frozen snapshot) preserved as-is.
4. **Best-effort external operations**: External provider failures are logged but
   never block the main conversation flow. Native memory always works regardless.
5. **Directory-based discovery**: Built-in providers in `deerflow/memory_providers/`,
   user-installed providers in `{base_dir}/plugins/memory/<name>/`.

### Conversation Lifecycle

```
Agent created
    ↓
MemoryManager.initialize_all(session_id)
    ↓
System prompt assembly:
    native.system_prompt_block() + external.system_prompt_block()
    ↓
Per-turn loop:
    prefetch_all(query) → external context (optional)
    ↓
    Agent generates response (may call provider tools)
    ↓
    handle_tool_call() → route to owning provider
    ↓
    sync_all(user, assistant) → persist turn
    ↓
Session ends:
    shutdown_all()
```

---

## 5.2 MemoryProvider Abstract Interface

```python
class MemoryProvider(ABC):
    """Pluggable memory backend interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'native', 'mem0', 'honcho', etc."""

    # --- Lifecycle ---

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and ready."""

    def initialize(self, session_id: str, **kwargs) -> None:
        """One-time setup per session. Default: no-op."""

    def shutdown(self) -> None:
        """Clean shutdown. Default: no-op."""

    # --- System Prompt ---

    def system_prompt_block(self) -> str:
        """Static text injected into system prompt. Default: empty string."""

    # --- Per-turn ---

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall context before each turn. Default: empty string."""

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        """Persist completed turn. Default: no-op."""

    # --- Tools ---

    def get_tool_schemas(self) -> list[dict]:
        """Return LangChain tool schemas for this provider. Default: []."""

    def handle_tool_call(self, tool_name: str, args: dict,
                         **kwargs) -> str:
        """Dispatch tool call. Default: raise ValueError."""
        raise ValueError(f"Unknown tool: {tool_name}")
```

### NativeMemoryProvider Adapter

Wraps existing `MemoryStore`:

- `name` → `"native"`
- `is_available()` → `True` always
- `system_prompt_block()` → Returns formatted MEMORY.md + USER.md frozen snapshot
  (delegates to `store.format_for_system_prompt()`)
- `get_tool_schemas()` → Returns the existing `memory` tool schema (add/replace/remove)
- `handle_tool_call()` → Delegates to the logic in `create_memory_tool()`
- `prefetch()` / `sync_turn()` → No-op (native store doesn't need per-turn sync)

---

## 5.3 MemoryManager Orchestrator

```python
class MemoryManager:
    """Orchestrates native + optional external memory provider."""

    def __init__(self, native: MemoryProvider):
        self._native = native
        self._external: MemoryProvider | None = None
        self._tool_to_provider: dict[str, MemoryProvider] = {}
        self._initialized = False

    def add_provider(self, provider: MemoryProvider) -> None:
        """Register external provider. Raises if one already registered."""
        if self._external is not None:
            raise ValueError(
                f"Already have external provider '{self._external.name}', "
                f"cannot add '{provider.name}'"
            )
        self._external = provider

    def initialize_all(self, session_id: str, **kwargs) -> None:
        """Initialize all providers. External failures logged, not raised."""
        self._native.initialize(session_id, **kwargs)
        if self._external:
            try:
                self._external.initialize(session_id, **kwargs)
            except Exception as exc:
                logger.warning(
                    "External provider '%s' init failed: %s",
                    self._external.name, exc,
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

    def build_system_prompt(self) -> str:
        """Collect system prompt blocks from all providers."""
        blocks = [self._native.system_prompt_block()]
        if self._external:
            ext_block = self._external.system_prompt_block()
            if ext_block:
                blocks.append(ext_block)
        return "\n\n".join(b for b in blocks if b)

    def build_external_prompt(self) -> str:
        """Return external provider's prompt block only (for prompt.py)."""
        if self._external:
            return self._external.system_prompt_block()
        return ""

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Collect prefetch context. External failures return empty."""
        results = []
        if self._external:
            try:
                ctx = self._external.prefetch(query, session_id=session_id)
                if ctx:
                    results.append(ctx)
            except Exception as exc:
                logger.warning("External prefetch failed: %s", exc)
        return "\n\n".join(results)

    def sync_all(self, user_content: str, assistant_content: str,
                 *, session_id: str = "") -> None:
        """Sync turn to all providers. Best-effort for external."""
        if self._external:
            try:
                self._external.sync_turn(
                    user_content, assistant_content, session_id=session_id,
                )
            except Exception as exc:
                logger.warning("External sync_turn failed: %s", exc)

    def get_all_tool_schemas(self) -> list[dict]:
        """Collect tool schemas from all providers."""
        schemas = self._native.get_tool_schemas()
        if self._external:
            schemas.extend(self._external.get_tool_schemas())
        return schemas

    def handle_tool_call(self, tool_name: str, args: dict,
                         **kwargs) -> str:
        """Route tool call to owning provider."""
        provider = self._tool_to_provider.get(tool_name)
        if not provider:
            raise ValueError(f"Unknown tool: {tool_name}")
        return provider.handle_tool_call(tool_name, args, **kwargs)

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
```

---

## 5.4 Provider Discovery and Loading

### Directory Structure

```
deerflow/memory_providers/
├── __init__.py          # discover_providers(), load_provider()
├── provider.py          # MemoryProvider ABC
├── manager.py           # MemoryManager orchestrator
├── native.py            # NativeMemoryProvider (wraps MemoryStore)
└── mem0/
    ├── __init__.py      # Mem0Provider class
    └── tool.py          # Mem0 tool schemas & handlers
```

User-installed providers: `{base_dir}/plugins/memory/<name>/`

### Discovery Functions

```python
def discover_providers(base_dir: Path | None = None) -> list[tuple[str, str, bool]]:
    """Return (name, module_path, is_builtin) for each found provider.

    Scan order:
    1. Built-in: deerflow.memory_providers.<name> subdirectories
    2. User-installed: {base_dir}/plugins/memory/<name>/
    Built-in takes precedence on name collision.
    """

def load_provider(name: str, base_dir: Path | None = None) -> MemoryProvider | None:
    """Load and instantiate provider by name.

    1. Check built-in providers first
    2. Fall back to user-installed
    3. Import module, find MemoryProvider subclass, instantiate
    4. Return None if not found or import fails
    """
```

### Configuration

New file: `config/memory_provider_config.py`

```python
class MemoryProviderConfig(BaseModel):
    enabled: bool = False
    name: str = ""  # Provider name: "mem0", "honcho", etc. Empty = native only
```

In `config/app_config.py`:
```python
memory_provider: MemoryProviderConfig = Field(
    default_factory=MemoryProviderConfig,
    description="External memory provider configuration",
)
```

In `config.example.yaml`:
```yaml
# External memory provider (Mem0, Honcho, etc.)
memory_provider:
  enabled: false
  name: ""  # "mem0" to enable Mem0 provider
```

### Loading Flow

1. Read `memory_provider.enabled` and `memory_provider.name`
2. `enabled=false` or `name=""` → native-only, skip external loading
3. `enabled=true` and `name` set → `load_provider(name, base_dir)`
4. Provider's `is_available()` returns `False` → log warning, degrade to native-only
5. Provider instantiated and registered with `MemoryManager.add_provider()`

---

## 5.5 Mem0 Provider (Example Implementation)

### Dependencies

- `mem0ai` Python package (optional dependency)
- `MEM0_API_KEY` environment variable

### Tools

**`mem0_search`** — Semantic search over stored memories:
```python
{
    "name": "mem0_search",
    "description": "Search your Mem0 long-term memory for relevant context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
}
```

**`mem0_add`** — Manually add a memory:
```python
{
    "name": "mem0_add",
    "description": "Add a specific memory to your Mem0 store.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Content to remember"},
            "metadata": {"type": "object", "description": "Optional metadata"},
        },
        "required": ["content"],
    },
}
```

### Implementation

```python
class Mem0Provider(MemoryProvider):
    """Mem0 external memory provider."""

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
            from mem0 import MemoryClient
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        from mem0 import MemoryClient
        self._client = MemoryClient(api_key=self._api_key)
        self._user_id = kwargs.get("user_id", self._user_id or session_id)

    def system_prompt_block(self) -> str:
        return (
            "You have access to a Mem0 long-term memory service.\n"
            "Use mem0_search to find relevant past context.\n"
            "Use mem0_add to store important information."
        )

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

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
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
    CIRCUIT_THRESHOLD = 5
    CIRCUIT_COOLDOWN = 300  # 5 minutes

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
```

---

## 5.6 Integration

### Shared State

The `MemoryManager` instance is created in `_make_lead_agent()` and stored in the
agent's `config` dict so it's accessible to both the tool registration and the prompt
builder:

```python
config["memory_manager"] = memory_manager
```

This mirrors how other shared objects (e.g., `thread_id`) are passed through the config dict.

### Lead Agent Tool Registration

In `agents/lead_agent/agent.py`, within the existing `memory.enabled` block, add
MemoryManager creation after the existing `MemoryStore` setup:

```python
if resolved_app_config.memory.enabled:
    # ... existing MemoryStore creation code (store = MemoryStore(...)) ...

    # NEW: Wrap store in MemoryManager
    from deerflow.memory_providers.manager import MemoryManager
    from deerflow.memory_providers.native import NativeMemoryProvider
    from deerflow.memory_providers import load_provider

    native_provider = NativeMemoryProvider(store=store)
    memory_manager = MemoryManager(native_provider)

    # Load external provider if configured
    if (resolved_app_config.memory_provider.enabled
            and resolved_app_config.memory_provider.name):
        ext = load_provider(
            resolved_app_config.memory_provider.name,
            base_dir=paths.base_dir,
        )
        if ext and ext.is_available():
            memory_manager.add_provider(ext)
            logger.info("Loaded external memory provider: %s", ext.name)
        else:
            logger.warning(
                "Provider '%s' not available, using native only",
                resolved_app_config.memory_provider.name,
            )

    memory_manager.initialize_all(thread_id)
    config["memory_manager"] = memory_manager

    # Register native memory tool (unchanged)
    extra_tools.append(create_memory_tool(store))

    # Register external provider tools (if any)
    for schema in memory_manager.get_all_tool_schemas():
        if schema["name"] != "memory":  # native tool already registered above
            extra_tools.append(
                _create_routed_tool(memory_manager, schema)
            )
```

Note: The native `memory` tool continues to use `create_memory_tool(store)` as before.
Only external provider tools go through `_create_routed_tool`.

### System Prompt Injection

In `agents/lead_agent/prompt.py`, append external provider block to the existing
`_get_memory_context` function. The `memory_manager` is retrieved from the agent's
config dict:

```python
def _get_memory_context(agent_name, app_config, memory_manager=None):
    # Existing native memory block logic unchanged
    native_block = _format_native_memory(store)

    # Append external provider block if MemoryManager has external provider
    if memory_manager:
        ext_block = memory_manager.build_external_prompt()
        if ext_block:
            native_block += f"\n\n{ext_block}"

    return native_block
```

The `memory_manager` parameter is added to the function signature. The caller in
`prompt.py` reads it from `config.get("memory_manager")` and passes it through.

### Per-turn Sync

In `agents/middlewares/memory_nudge_middleware.py` or a new lightweight middleware,
add `sync_all()` after each assistant response:

```python
# After each assistant response is complete
memory_manager = config.get("memory_manager")
if memory_manager:
    memory_manager.sync_all(
        user_content=user_msg,
        assistant_content=assistant_msg,
        session_id=thread_id,
    )
```

This can be added to the existing `MemoryNudgeMiddleware.on_after_agent()` hook or
as a separate `MemorySyncMiddleware`.

### Tool Routing Helper

```python
def _create_routed_tool(manager: MemoryManager, schema: dict):
    """Create a LangChain tool that routes through MemoryManager."""
    tool_name = schema["name"]

    @tool(tool_name)
    def routed_tool(query: str = "", content: str = "", metadata: dict = None) -> str:
        # Collect non-None, non-empty args
        args = {k: v for k, v in locals().items()
                if v is not None and v != "" and k != "tool_name"}
        return manager.handle_tool_call(tool_name, args)

    routed_tool.description = schema.get("description", "")
    return routed_tool
```

---

## 5.7 Files Summary

### New Files

| File | Lines (est.) | Description |
|------|-------------|-------------|
| `deerflow/memory_providers/__init__.py` | ~60 | discover_providers(), load_provider() |
| `deerflow/memory_providers/provider.py` | ~50 | MemoryProvider ABC |
| `deerflow/memory_providers/manager.py` | ~100 | MemoryManager orchestrator |
| `deerflow/memory_providers/native.py` | ~80 | NativeMemoryProvider adapter |
| `deerflow/memory_providers/mem0/__init__.py` | ~120 | Mem0Provider implementation |
| `deerflow/memory_providers/mem0/tool.py` | ~60 | Mem0 tool schemas and handlers |
| `config/memory_provider_config.py` | ~20 | Configuration model |

### Modified Files

| File | Change |
|------|--------|
| `config/app_config.py` | Add `memory_provider: MemoryProviderConfig` field |
| `agents/lead_agent/agent.py` | Register tools through MemoryManager |
| `agents/lead_agent/prompt.py` | Append external provider system_prompt_block |
| `config.example.yaml` | Add `memory_provider` config section |

### New Test Files

| File | Tests |
|------|-------|
| `tests/test_memory_provider_abc.py` | ABC interface, default implementations |
| `tests/test_memory_manager.py` | Orchestration, tool routing, failure isolation |
| `tests/test_native_provider.py` | MemoryStore wrapper adapter |
| `tests/test_mem0_provider.py` | Mem0 lifecycle, tools, circuit breaker |

---

## 5.8 Error Handling

- **External provider init failure**: Logged as warning, degrades to native-only
- **prefetch/sync_turn failure**: Best-effort, logged, never blocks conversation
- **Tool call failure**: Error message returned to agent, no crash
- **Provider not found**: Config error logged, uses native-only
- **Mem0 API consecutive failures**: Circuit breaker opens (threshold 5), closes after
  5-minute cooldown

---

## 5.9 Out of Scope

- Multiple simultaneous external providers
- Hermes multi-session hooks (on_delegation, on_session_switch, on_pre_compress)
- Provider setup wizard / CLI configuration flow
- Streaming context scrubbing for external provider output (can add later if needed)
- Additional built-in providers (Honcho, SuperMemory, etc.) — community can add via plugin
- Async provider methods (sync API sufficient for initial implementation)
