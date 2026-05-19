# Phase 3 Design: Memory Nudge + Streaming Scrubber

Hermes-style memory migration — Phase 3 of the DeerFlow memory architecture alignment.

## Overview

Phase 3 adds two features to the Hermes-style memory system implemented in Phase 1-2:

1. **Memory Nudge**: Periodic background review that reminds the agent to save memories
2. **Streaming Scrubber**: Safety net that strips `<memory>` tags from streaming output to prevent leakage

Both features are faithful ports from Hermes, adapted to DeerFlow's LangGraph/LangChain architecture.

---

## 3.1 Memory Nudge

### Problem

The Hermes-style memory system relies on the agent proactively using the `memory` tool to save information. Without prompts, the agent may forget to save important user preferences or context revealed during conversation.

### Solution

A middleware that counts conversation turns and spawns a background task to review the conversation when the threshold is reached. The background task creates a lightweight agent with memory tools and asks it to review the conversation for save-worthy information.

### Architecture

```
User turn → LLM response → after_model hook (MemoryNudgeMiddleware)
                                    ↓
                          Turn counter incremented
                                    ↓
                          Threshold reached?
                            ↓ Yes          ↓ No
                    asyncio.create_task   Continue
                            ↓
                    Background review agent
                    (LLM + memory tool only)
                            ↓
                    Reviews conversation
                    Decides what to save
                            ↓
                    Writes via memory tool
                    (logged, not injected)
```

### New File: `packages/harness/deerflow/agents/middlewares/memory_nudge_middleware.py`

```python
class MemoryNudgeMiddleware(AgentMiddleware):
    """Periodically spawns a background memory review."""

    def __init__(self, *, app_config: AppConfig | None = None):
        super().__init__()
        self._app_config = app_config
        self._last_triggered_count = 0

    def after_model(self, state, runtime) -> dict | None:
        # Count user messages in state
        # If count >= nudge_interval and count % interval == 0
        #   and count != last_triggered_count:
        #   Spawn background task via asyncio.get_running_loop().create_task()
        #   Update last_triggered_count
        return None

    async def _run_background_review(self, messages, store):
        # Create lightweight agent: model + memory tool
        # Feed review prompt + last N messages
        # Catch and log all exceptions
```

**Review prompt** (ported from Hermes):
```
Review the conversation above and consider saving to memory if appropriate.

Focus on:
1. Has the user revealed things about themselves — their persona, desires,
   preferences, or personal details worth remembering?
2. Has the user expressed expectations about how you should behave, their work
   style, or ways they want you to operate?

If something stands out, save it using the memory tool.
If nothing is worth saving, just say 'Nothing to save.' and stop.
```

**Background agent creation**:
- Uses `create_chat_model()` for the LLM
- Uses `create_memory_tool(store)` for the memory tool
- Uses LangChain's `create_agent()` directly (not LangGraph) — lightweight, no middlewares, no sandbox
- Only the last 20 messages are included to keep context manageable

### Config Change: `packages/harness/deerflow/config/memory_config.py`

Add field:
```python
nudge_interval: int = 10  # Number of user turns between reviews; 0 = disabled
```

### Integration: `packages/harness/deerflow/agents/lead_agent/agent.py`

In `_build_middlewares()`, after TitleMiddleware:
```python
if resolved_app_config.memory.enabled and resolved_app_config.memory.nudge_interval > 0:
    middlewares.append(MemoryNudgeMiddleware(app_config=resolved_app_config))
```

### Key Design Decisions

1. **Why middleware, not worker hook**: Middlewares have access to state (messages) and runtime context. Worker hooks don't have the same visibility.
2. **Why lightweight agent, not full lead agent**: The review only needs LLM + memory tool. Full lead agent includes sandbox, MCP, subagents, and all middlewares — unnecessary overhead.
3. **Why asyncio.create_task**: The middleware's `after_model` runs inside LangGraph's event loop. `asyncio.get_running_loop().create_task()` is the correct way to spawn background work from a sync context.
4. **Why track `_last_triggered_count`**: Prevents re-triggering when the same turn count is seen multiple times (e.g., after tool calls within the same turn).

---

## 3.2 Streaming Memory Scrubber

### Problem

If the LLM accidentally outputs `<memory>` tags in its response (echoing back injected memory context), these tags would be visible to the user via SSE streaming. Simple regex replacement fails when tags are split across streaming chunks.

### Solution

A stateful scrubber that tracks whether we're inside a `<memory>` span and holds back partial tags in an internal buffer. Ported from Hermes's `StreamingContextScrubber`, adapted for `<memory>` tags.

### Architecture

```
LLM delta chunk → worker.py extract AIMessageChunk.content
                        ↓
                scrubber.feed(text) → scrubbed text
                        ↓
                Update chunk.content
                        ↓
                serialize() → bridge.publish() → SSE
```

State machine:
```
OUTSIDE ──"<memory>"──→ INSIDE ──"</memory>"──→ OUTSIDE
   │                       │
   ↓ (hold partial tag)    ↓ (discard all content)
```

### New File: `packages/harness/deerflow/agents/memory/scrubber.py`

```python
class StreamingMemoryScrubber:
    """Stateful scrubber for streaming text containing <memory> tags.

    Handles tags split across delta chunks by maintaining an internal
    buffer for partial tag matches.
    """

    _OPEN_TAG = "<memory>"
    _CLOSE_TAG = "</memory>"

    def __init__(self):
        self._in_span: bool = False
        self._buf: str = ""

    def reset(self) -> None:
        self._in_span = False
        self._buf = ""

    def feed(self, text: str) -> str:
        """Return visible portion of text after scrubbing."""
        # Hold back partial tag suffixes
        # Discard everything inside <memory> spans
        # Emit text outside spans
        ...

    def flush(self) -> str:
        """Emit any held-back buffer at end-of-stream."""
        # If still in span, discard (safer than leaking)
        # If partial tag that wasn't real, emit it
        ...

    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        """Return length of longest buf-suffix that could start the tag."""
        ...
```

### Integration: `packages/harness/deerflow/runtime/runs/worker.py`

In `run_agent()`:

1. **Initialization** (after agent creation):
```python
scrubber = StreamingMemoryScrubber()
```

2. **Per-chunk scrubbing** (in the streaming loops):
```python
# For messages mode, scrub AIMessageChunk content
if mode == "messages" and chunk is AIMessageChunk:
    text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
    scrubbed = scrubber.feed(text)
    if scrubbed != text:
        chunk = chunk.model_copy(update={"content": scrubbed})
```

3. **End of stream** (in finally block):
```python
scrubber.flush()
```

### Key Design Decisions

1. **Why scrub in worker.py**: The streaming pipeline flows through worker → serialize → bridge. Middlewares operate on state/tool calls, not SSE output. Worker is the only place with access to individual message chunks before serialization.
2. **Why only AIMessageChunk**: HumanMessage and ToolMessage are not at risk of leaking memory context. Only the LLM's output could contain echoed memory tags.
3. **Why stateful, not regex**: When a `<memory>` tag is split across chunks (e.g., `<memo` in one delta, `ry>content</memory>` in the next), a simple regex on each chunk would fail to match the tag.
4. **Safe-by-default flush**: If the stream ends while inside an open span, the remaining buffer is discarded. Leaking partial memory context is worse than truncating output.

---

## Files Summary

### New Files

| File | Lines (est.) | Description |
|------|-------------|-------------|
| `agents/middlewares/memory_nudge_middleware.py` | ~100 | Nudge middleware with background review |
| `agents/memory/scrubber.py` | ~120 | StreamingMemoryScrubber state machine |

### Modified Files

| File | Change |
|------|--------|
| `config/memory_config.py` | Add `nudge_interval: int = 10` field |
| `agents/memory/__init__.py` | Export `StreamingMemoryScrubber` |
| `agents/lead_agent/agent.py` | Add `MemoryNudgeMiddleware` to chain |
| `runtime/runs/worker.py` | Integrate scrubber into streaming pipeline |
| `config.example.yaml` | Add `nudge_interval` to memory config section |

### New Test Files

| File | Tests |
|------|-------|
| `tests/test_memory_nudge_middleware.py` | Turn counting, threshold triggering, background task spawning, disabled when interval=0 |
| `tests/test_streaming_memory_scrubber.py` | Tag detection, split tags, nested tags, partial tag buffering, flush behavior |

---

## Config Changes

`config.yaml` memory section gains one new field:

```yaml
memory:
  enabled: true
  injection_enabled: true
  storage_path: ".deer-flow"
  memory_char_limit: 2200
  user_char_limit: 1375
  nudge_interval: 10  # NEW: user turns between background memory reviews; 0 = disabled
```

---

## Error Handling

- **Background review failure**: Logged as warning, never crashes the main conversation
- **Scrubber error**: If scrubbing fails, original text is passed through (fail-open for output, fail-closed for security-sensitive spans)
- **Agent creation failure in background**: Logged, task silently ends

---

## Out of Scope

- Counter hydration for gateway reconnection (can be added later if needed)
- `<memory-context>` tag support (not used in DeerFlow)
- Integration with IM channels (channels use `runs.wait()` not streaming, scrubber applies to SSE mode only)
