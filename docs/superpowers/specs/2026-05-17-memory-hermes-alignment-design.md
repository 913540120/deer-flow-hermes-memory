# Phase 1: Memory System Hermes Alignment — Design Spec

**Date**: 2026-05-17
**Status**: Approved
**Approach**: Direct port from Hermes `memory_tool.py`, remove existing LLM auto-extraction pipeline

## Goal

Replace DeerFlow's current JSON-based auto-extraction memory system with Hermes's Agent-driven curated memory system. The Agent actively manages persistent memory via a `memory` tool (add/replace/remove) writing to MEMORY.md and USER.md files. A frozen snapshot pattern preserves prefix cache stability.

No innovation — faithful reproduction of Hermes behavior.

## Design Decisions

- **Complete replacement**: Remove all auto-extraction code (MemoryUpdater, MemoryUpdateQueue, MemoryMiddleware). The only way memory gets written is through the Agent's explicit tool calls.
- **Direct port**: Copy Hermes's MemoryStore class and security scanner, adapting only directory paths and import structure to DeerFlow conventions.
- **Per-user isolation preserved**: DeerFlow's existing `{base_dir}/users/{user_id}/` pattern is kept. Memory files go under `{base_dir}/users/{user_id}/memory/MEMORY.md` and `USER.md`.
- **Character limits** (not tokens): Port Hermes's character-based limits (MEMORY.md: 2200 chars, USER.md: 1375 chars) for model independence.
- **Frozen snapshot**: At session start, load files and capture a snapshot. System prompt injection uses this snapshot for the entire session. Tool writes update disk immediately but don't affect the snapshot.

## File Changes

### New Files

#### `packages/harness/deerflow/agents/memory/store.py` (~300 lines)

Core `MemoryStore` class, ported from Hermes's `tools/memory_tool.py`.

```python
class MemoryStore:
    """
    Bounded curated memory with file persistence.
    Two targets: "memory" (agent notes) and "user" (user profile).
    Frozen snapshot pattern for prefix cache stability.
    """
    ENTRY_DELIMITER = "\n\u00a7\n"  # section sign

    def __init__(self, memory_dir: Path, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_dir = memory_dir
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self._system_prompt_snapshot: dict[str, str] = {"memory": "", "user": ""}

    # --- Lifecycle ---
    def load_from_disk(self) -> None:
        """Load entries from MEMORY.md and USER.md, capture frozen snapshot."""

    # --- Write Operations ---
    def add(self, target: str, content: str) -> dict[str, Any]:
        """Append entry. Security scan → lock → dedup → limit check → atomic write."""

    def replace(self, target: str, old_text: str, new_content: str) -> dict[str, Any]:
        """Find entry by substring match, replace it."""

    def remove(self, target: str, old_text: str) -> dict[str, Any]:
        """Remove entry matching substring."""

    # --- Read Operations ---
    def format_for_system_prompt(self, target: str) -> str | None:
        """Return frozen snapshot for system prompt injection (NOT live state)."""

    def get_live_entries(self, target: str) -> list[str]:
        """Return current live entries (for tool response)."""

    # --- Internal ---
    @staticmethod
    @contextmanager
    def _file_lock(path: Path): ...

    def _reload_target(self, target: str): ...
    def save_to_disk(self, target: str): ...
    @staticmethod
    def _read_file(path: Path) -> list[str]: ...
    @staticmethod
    def _write_file(path: Path, entries: list[str]): ...
    def _render_block(self, target: str, entries: list[str]) -> str: ...
    def _entries_for(self, target: str) -> list[str]: ...
    def _set_entries(self, target: str, entries: list[str]): ...
    def _char_count(self, target: str) -> int: ...
    def _char_limit(self, target: str) -> int: ...
```

Key behaviors ported from Hermes:
- Entry delimiter: `\n§\n` (section sign)
- File locking: `fcntl.flock` on Unix, `msvcrt.locking` on Windows, using separate `.lock` sidecar file
- Atomic writes: `tempfile.mkstemp` + `os.fsync` + `os.replace` in same directory
- Lock-then-reload: Under file lock, re-read from disk to pick up concurrent writes from other sessions
- Duplicate detection: Exact string match prevents duplicate entries
- Substring matching for replace/remove: Short unique substring identifies the target entry; rejects ambiguous matches unless all matches are identical
- Character budget enforcement: Adding/replacing entries that would exceed the limit returns an error with current usage info

#### `packages/harness/deerflow/agents/memory/security.py` (~80 lines)

Memory content threat scanner, ported from Hermes.

```python
_MEMORY_THREAT_PATTERNS: list[tuple[str, str]] = [
    # (compiled_regex_pattern, pattern_id)
    # Covers: prompt injection, role hijacking, deception, exfiltration via curl/wget,
    #         reading secrets, SSH backdoor, .hermes/.env access
]

_INVISIBLE_CHARS: set[str] = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}

def scan_memory_content(content: str) -> str | None:
    """
    Scan memory content for injection/exfiltration patterns.
    Returns error string if blocked, None if safe.
    Checks: invisible unicode characters + regex threat patterns.
    """
```

#### `packages/harness/deerflow/agents/memory/tool.py` (~80 lines)

Memory tool definition and handler for DeerFlow's agent tool system.

```python
MEMORY_TOOL_SCHEMA = {
    "name": "memory",
    "description": "<ported from Hermes MEMORY_SCHEMA description>",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
        },
        "required": ["action", "target"],
    },
}

def create_memory_tool(store: MemoryStore) -> dict:
    """Create memory tool definition with store bound."""

async def handle_memory_tool(args: dict, store: MemoryStore) -> str:
    """Handle memory tool invocation. Returns JSON string."""
```

### Modified Files

#### `packages/harness/deerflow/agents/memory/prompt.py`

**Remove**: `MEMORY_UPDATE_PROMPT`, `FACT_EXTRACTION_PROMPT`, `format_conversation_for_update`, `_count_tokens`, `_coerce_confidence`, `format_memory_for_injection` (all auto-extraction related).

**Keep/Adapt**: A simple helper that takes the snapshot strings and wraps them for system prompt injection:

```python
def format_memory_block(memory_snapshot: str | None, user_snapshot: str | None) -> str:
    """Format frozen memory snapshots for system prompt injection.
    Returns empty string if both snapshots are empty."""
```

#### `packages/harness/deerflow/agents/memory/__init__.py`

Update exports to expose `MemoryStore`, `scan_memory_content`, and tool functions. Remove exports for deleted modules (`MemoryUpdater`, `MemoryUpdateQueue`, etc.).

#### `packages/harness/deerflow/config/memory_config.py`

Replace with Hermes-style configuration:

```python
class MemoryConfig(BaseModel):
    enabled: bool = True
    injection_enabled: bool = True
    storage_path: str = "~/.deerflow/memory"
    memory_char_limit: int = 2200
    user_char_limit: int = 1375
```

Removed fields: `debounce_seconds`, `max_facts`, `fact_confidence_threshold`, `max_injection_tokens`, `model_name` (no longer needed without auto-extraction).

#### `packages/harness/deerflow/agents/lead_agent/agent.py`

Changes:
1. Remove `from ..middlewares.memory_middleware import MemoryMiddleware` import
2. Remove `MemoryMiddleware(agent_name=agent_name)` from middleware chain
3. Remove `memory_flush_hook` registration from `SummarizationMiddleware`
4. Add `memory` tool to available tools (conditionally, when `MemoryConfig.enabled`)
5. Create `MemoryStore` instance during agent initialization, pass to tool handler

#### `packages/harness/deerflow/agents/lead_agent/prompt.py`

Changes to `_get_memory_context()`:
1. Load `MemoryStore` instance for the current user
2. Call `store.format_for_system_prompt("memory")` and `store.format_for_system_prompt("user")`
3. Concatenate snapshots into the system prompt memory section
4. Remove all JSON-based formatting logic

### Deleted Files

| File | Lines | Reason |
|------|-------|--------|
| `memory/updater.py` | 612 | LLM auto-extraction replaced by Agent tool |
| `memory/queue.py` | 278 | Debounce queue no longer needed |
| `memory/message_processing.py` | 110 | Message filtering only served auto-extraction |
| `memory/summarization_hook.py` | 32 | Flush hook only served auto-extraction queue |
| `middlewares/memory_middleware.py` | 111 | Middleware only served auto-extraction queue |

### Files to check for references to deleted modules

Before deletion, search the entire codebase for imports/references to:
- `MemoryUpdater`, `update_memory_from_conversation`, `get_memory_data`
- `MemoryUpdateQueue`, `get_memory_queue`
- `MemoryMiddleware`
- `memory_flush_hook`
- `filter_messages_for_memory`, `detect_correction`
- `format_conversation_for_update`, `MEMORY_UPDATE_PROMPT`

All references must be removed or redirected.

## Data Flow

### Memory Write (Agent Tool)

```
Agent decides to save memory
  → calls memory tool (action=add, target=memory, content="...")
  → handle_memory_tool()
  → security.scan_memory_content(content)
  → MemoryStore.add("memory", content)
      → _file_lock()
      → _reload_target() — pick up concurrent writes
      → duplicate check
      → character limit check
      → _write_file() — tempfile + fsync + atomic rename
      → update live entries
  → return JSON result to Agent (shows live state, not snapshot)
```

### Memory Read (System Prompt Injection)

```
Agent turn starts
  → apply_prompt_template()
  → _get_memory_context()
  → store.format_for_system_prompt("memory")  → returns frozen snapshot string
  → store.format_for_system_prompt("user")    → returns frozen snapshot string
  → format_memory_block(memory_snapshot, user_snapshot)
  → injected into system prompt at {memory_context} placeholder
```

### Session Lifecycle

```
Session/Thread creation:
  → MemoryStore.__init__(memory_dir, char_limits)
  → store.load_from_disk()
      → read MEMORY.md → memory_entries
      → read USER.md → user_entries
      → _system_prompt_snapshot = {
             "memory": _render_block("memory", memory_entries),
             "user": _render_block("user", user_entries),
         }

During session:
  → Tool writes → update disk immediately, live entries updated
  → System prompt injection → always uses frozen snapshot
  → Tool responses → show live entries (so Agent sees its writes)

Session/Thread end:
  → MemoryStore instance released
  → Next session → fresh load_from_disk() picks up any writes
```

## Snapshot Rendering Format

Ported from Hermes's `_render_block`:

```
════════════════════════════════════════════════
MEMORY (your personal notes) [45% — 990/2,200 chars]
════════════════════════════════════════════════
User prefers TypeScript over JavaScript for project development
§
Project deployed on Alibaba Cloud ECS with Docker Compose
§
Must run pnpm check before every commit or CI fails
```

```
════════════════════════════════════════════════
USER PROFILE (who the user is) [30% — 412/1,375 chars]
════════════════════════════════════════════════
Native Chinese speaker, fluent in English
§
Prefers concise communication style, no unnecessary pleasantries
§
Full-stack developer, primarily TypeScript and Python
```

## Error Handling

- **Security scan failure**: Tool returns `{"success": false, "error": "Blocked: ..."}` — Agent sees the rejection and can inform the user.
- **Character limit exceeded**: Tool returns `{"success": false, "error": "Memory at X/Y chars...", "usage": "..."}` — Agent knows it needs to replace/remove first.
- **Ambiguous match (replace/remove)**: Tool returns `{"success": false, "error": "Multiple entries matched...", "matches": [...]}` — Agent can be more specific.
- **File I/O failure**: `MemoryStore._write_file` raises `RuntimeError` — caught by tool handler, returns error to Agent.
- **Memory not loaded**: If `load_from_disk()` was never called, `format_for_system_prompt` returns empty strings — graceful degradation.

## Storage Directory Layout

```
{storage_path}/
└── users/
    └── {user_id}/
        └── memory/
            ├── MEMORY.md          # Agent's personal notes
            ├── USER.md            # User profile
            ├── MEMORY.md.lock     # File lock sidecar (created on demand)
            └── USER.md.lock       # File lock sidecar (created on demand)
```

For per-agent memory (when `agent_name` is set):
```
{storage_path}/
└── users/
    └── {user_id}/
        └── agents/
            └── {agent_name}/
                └── memory/
                    ├── MEMORY.md
                    └── USER.md
```

## Configuration

`config.yaml` or environment variables:

```yaml
memory:
  enabled: true
  injection_enabled: true
  storage_path: ~/.deerflow/memory
  memory_char_limit: 2200
  user_char_limit: 1375
```

## API Layer Changes

### Gateway Memory Router (`app/gateway/routers/memory.py`)

This router currently exposes REST endpoints that operate on the JSON memory structure via `updater.py` functions. After migration, the endpoints need to work with the new `MemoryStore`.

**Current endpoints** (need adaptation):
- `GET /api/memory/` — returns full memory data (JSON structure)
- `POST /api/memory/reload` — force reload from disk
- `GET /api/memory/config` — return memory config
- `GET /api/memory/status` — return config + data summary

**Post-migration endpoints**:
- `GET /api/memory/` — return MEMORY.md + USER.md contents as structured JSON (entries list per target)
- `POST /api/memory/reload` — call `store.load_from_disk()` to refresh snapshot
- `GET /api/memory/config` — return new config (char limits, enabled flags)
- `GET /api/memory/status` — return config + entry counts + usage percentages
- `POST /api/memory/entries` — **new**: create/update/delete entries via API (optional, mirrors tool operations for UI)

### DeerFlowClient Memory Methods (`packages/harness/deerflow/client.py`)

Client methods at lines 853-1070 import from `updater.py`. After migration:

**Replace**:
- `get_memory()` → load via `MemoryStore`, return entries as JSON
- `reload_memory()` → `store.load_from_disk()`
- `get_memory_config()` → read from `MemoryConfig`
- `get_memory_status()` → config + store stats
- `create_memory_fact()` → `store.add()`
- `delete_memory_fact()` → `store.remove()`
- `update_memory_fact()` → `store.replace()`
- `clear_memory_data()` → clear both MEMORY.md and USER.md
- `import_memory_data()` → write entries to MEMORY.md / USER.md

### Test Files Impact

**Delete** (tests for removed modules):
- `tests/test_memory_updater.py`
- `tests/test_memory_queue.py`
- `tests/test_memory_queue_user_isolation.py`
- `tests/test_memory_updater_user_isolation.py`
- `tests/test_memory_upload_filtering.py`
- `tests/test_memory_prompt_injection.py` (tests `format_memory_for_injection` with JSON structure)

**Update** (tests that reference memory):
- `tests/test_memory_storage.py` — adapt to new `MemoryStore` API
- `tests/test_memory_storage_user_isolation.py` — adapt to new directory layout
- `tests/test_custom_agent.py` — adapt memory storage assertions
- `tests/test_summarization_middleware.py` — remove `memory_flush_hook` reference
- `tests/test_client.py` — adapt `TestGatewayConformance` memory response models

**New**:
- `tests/test_memory_store.py` — test `MemoryStore` add/replace/remove/frozen-snapshot/security
- `tests/test_memory_tool.py` — test `memory` tool handler

## Additional Files That Reference Memory

Full reference search found imports in:

| File | Action |
|------|--------|
| `agents/lead_agent/agent.py:8` | Remove `memory_flush_hook` import |
| `agents/lead_agent/agent.py:11` | Remove `MemoryMiddleware` import |
| `agents/lead_agent/agent.py:94-95` | Remove `memory_flush_hook` registration |
| `agents/lead_agent/agent.py:279` | Remove `MemoryMiddleware` from middleware chain |
| `agents/lead_agent/prompt.py:546` | Replace `format_memory_for_injection, get_memory_data` imports with new store |
| `agents/middlewares/memory_middleware.py` | Delete entire file |
| `agents/memory/__init__.py` | Rewrite exports |
| `app/gateway/routers/memory.py` | Rewrite to use `MemoryStore` |
| `client.py:853-1070` | Rewrite memory methods |
| `config/memory_config.py` | Replace with Hermes-style config |

## Summary of Code Size Impact

| Category | Lines |
|----------|-------|
| New: `store.py` | ~300 |
| New: `security.py` | ~80 |
| New: `tool.py` | ~80 |
| Modified: `prompt.py` | net -200 |
| Modified: `__init__.py` | ~20 |
| Modified: `memory_config.py` | net -30 |
| Modified: `agent.py` | ~30 |
| Modified: `prompt.py` (lead_agent) | ~20 |
| Deleted: `updater.py` | -612 |
| Deleted: `queue.py` | -278 |
| Deleted: `message_processing.py` | -110 |
| Deleted: `summarization_hook.py` | -32 |
| Deleted: `memory_middleware.py` | -111 |
| **Net change** | **~-740 lines** |
