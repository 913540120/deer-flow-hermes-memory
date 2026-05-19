# Memory Nudge + Streaming Scrubber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Hermes's Memory Nudge (background review) and StreamingContextScrubber to DeerFlow's LangGraph/LangChain architecture.

**Architecture:** Two independent features. (1) A `MemoryNudgeMiddleware` that counts turns via `after_model` and spawns background asyncio tasks with lightweight agents to review conversations for save-worthy memories. (2) A `StreamingMemoryScrubber` state machine that strips `<memory>` tags from streaming AIMessageChunk deltas in `worker.py` before they reach the SSE bridge.

**Tech Stack:** Python 3.12+, LangChain/LangGraph middleware, asyncio, pydantic, pytest.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `packages/harness/deerflow/agents/memory/scrubber.py` | StreamingMemoryScrubber state machine |
| Create | `packages/harness/deerflow/agents/middlewares/memory_nudge_middleware.py` | Nudge middleware with background review |
| Create | `tests/test_streaming_memory_scrubber.py` | Scrubber unit tests |
| Create | `tests/test_memory_nudge_middleware.py` | Nudge middleware unit tests |
| Modify | `packages/harness/deerflow/config/memory_config.py:10-17` | Add `nudge_interval` field |
| Modify | `packages/harness/deerflow/agents/memory/__init__.py` | Export `StreamingMemoryScrubber` |
| Modify | `packages/harness/deerflow/agents/lead_agent/agent.py:292-293` | Add nudge middleware to chain |
| Modify | `packages/harness/deerflow/runtime/runs/worker.py:271-298` | Integrate scrubber into streaming loops |
| Modify | `config.example.yaml:787-792` | Add `nudge_interval` to memory config |

---

### Task 1: StreamingMemoryScrubber — Tests

**Files:**
- Create: `tests/test_streaming_memory_scrubber.py`

- [ ] **Step 1: Write all scrubber tests**

```python
"""Tests for StreamingMemoryScrubber state machine."""

import pytest

from deerflow.agents.memory.scrubber import StreamingMemoryScrubber


class TestStreamingMemoryScrubberBasic:
    def test_clean_text_passes_through(self):
        s = StreamingMemoryScrubber()
        assert s.feed("Hello world") == "Hello world"

    def test_empty_string_returns_empty(self):
        s = StreamingMemoryScrubber()
        assert s.feed("") == ""

    def test_complete_memory_tag_removed(self):
        s = StreamingMemoryScrubber()
        text = "before<memory>secret data</memory>after"
        assert s.feed(text) == "beforeafter"

    def test_only_memory_tag_removed(self):
        s = StreamingMemoryScrubber()
        assert s.feed("<memory>everything</memory>") == ""

    def test_multiple_memory_tags_removed(self):
        s = StreamingMemoryScrubber()
        text = "a<memory>x</memory>b<memory>y</memory>c"
        assert s.feed(text) == "abc"

    def test_case_insensitive_tags(self):
        s = StreamingMemoryScrubber()
        text = "before<MEMORY>secret</MEMORY>after"
        assert s.feed(text) == "beforeafter"

    def test_multiline_content_removed(self):
        s = StreamingMemoryScrubber()
        text = "a<memory>line1\nline2\nline3</memory>b"
        assert s.feed(text) == "ab"


class TestStreamingMemoryScrubberSplit:
    def test_tag_split_across_two_feeds(self):
        s = StreamingMemoryScrubber()
        # First chunk ends with partial open tag
        assert s.feed("hello<memo") == "hello"
        # Second chunk completes the tag and closes it
        assert s.feed("ry>secret</memory>world") == "world"

    def test_close_tag_split_across_two_feeds(self):
        s = StreamingMemoryScrubber()
        assert s.feed("a<memory>secret") == "a"
        assert s.feed(" stuff</memo") == ""
        assert s.feed("ry>b") == "b"

    def test_tag_split_across_three_feeds(self):
        s = StreamingMemoryScrubber()
        assert s.feed("<mem") == ""
        assert s.feed("ory>hid") == ""
        assert s.feed("den</memory>visible") == "visible"

    def test_partial_tag_that_is_not_a_tag(self):
        s = StreamingMemoryScrubber()
        # "<memo" looks like a partial tag but the next chunk reveals it's not
        assert s.feed("hello<memo") == "hello"
        assert s.feed("randum>world") == "<memorandum>world"
        # Flush to release any held buffer
        assert s.flush() == ""

    def test_text_held_back_for_potential_tag(self):
        s = StreamingMemoryScrubber()
        # Feed text ending with partial tag start
        result = s.feed("hello<mem")
        assert result == "hello"
        # Feed continuation that makes it a real tag
        result = s.feed("ory>secret</memory> world")
        assert result == " world"


class TestStreamingMemoryScrubberFlush:
    def test_flush_emits_held_partial_tag(self):
        s = StreamingMemoryScrubber()
        s.feed("hello<mem")
        # The "<mem" is held back because it could be a partial tag
        assert s.flush() == "<mem"

    def test_flush_discards_inside_span(self):
        s = StreamingMemoryScrubber()
        s.feed("hello<memory>secret stuff")
        # Still inside a span — flush discards everything
        assert s.flush() == ""

    def test_flush_clears_state(self):
        s = StreamingMemoryScrubber()
        s.feed("hello<memory>secret")
        s.flush()
        # After flush, scrubber is clean
        assert s.feed("clean text") == "clean text"

    def test_flush_empty_buffer(self):
        s = StreamingMemoryScrubber()
        assert s.flush() == ""


class TestStreamingMemoryScrubberReset:
    def test_reset_clears_state(self):
        s = StreamingMemoryScrubber()
        s.feed("hello<memory>secret")
        s.reset()
        # After reset, scrubber is clean and outside any span
        assert s.feed("clean") == "clean"

    def test_reset_then_flush(self):
        s = StreamingMemoryScrubber()
        s.feed("<memo")
        s.reset()
        assert s.flush() == ""


class TestStreamingMemoryScrubberEdgeCases:
    def test_adjacent_tags(self):
        s = StreamingMemoryScrubber()
        assert s.feed("<memory>a</memory><memory>b</memory>") == ""

    def test_tag_at_start(self):
        s = StreamingMemoryScrubber()
        assert s.feed("<memory>hidden</memory>visible") == "visible"

    def test_tag_at_end(self):
        s = StreamingMemoryScrubber()
        assert s.feed("visible<memory>hidden</memory>") == "visible"

    def test_empty_memory_tag(self):
        s = StreamingMemoryScrubber()
        assert s.feed("before<memory></memory>after") == "beforeafter"

    def test_no_held_text_when_no_partial_tag(self):
        s = StreamingMemoryScrubber()
        assert s.feed("complete text") == "complete text"
        assert s.flush() == ""

    def test_open_tag_only_never_closed(self):
        s = StreamingMemoryScrubber()
        assert s.feed("a<memory>b") == "a"
        assert s.flush() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_streaming_memory_scrubber.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deerflow.agents.memory.scrubber'`

---

### Task 2: StreamingMemoryScrubber — Implementation

**Files:**
- Create: `packages/harness/deerflow/agents/memory/scrubber.py`

- [ ] **Step 1: Implement the scrubber**

```python
"""Stateful scrubber for streaming text containing <memory> tags.

Strips ``<memory>...</memory>`` spans from streaming text deltas.
Handles tags split across chunks by maintaining an internal buffer
for partial tag matches. Ported from Hermes StreamingContextScrubber.
"""


class StreamingMemoryScrubber:
    """Stateful scrubber for streaming text that may contain split memory spans.

    The scrubber runs a small state machine across deltas, holding back partial
    tag tails and discarding everything inside a ``<memory>`` span.
    """

    _OPEN_TAG = "<memory>"
    _CLOSE_TAG = "</memory>"

    def __init__(self) -> None:
        self._in_span: bool = False
        self._buf: str = ""

    def reset(self) -> None:
        """Reset to initial state."""
        self._in_span = False
        self._buf = ""

    def feed(self, text: str) -> str:
        """Return the visible portion of *text* after scrubbing.

        Any trailing fragment that could be the start of an open/close tag
        is held back in the internal buffer and surfaced on the next
        ``feed()`` call or discarded/emitted by ``flush()``.
        """
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_span:
                idx = buf.lower().find(self._CLOSE_TAG)
                if idx == -1:
                    # Hold back a potential partial close tag; drop the rest
                    held = self._max_partial_suffix(buf, self._CLOSE_TAG)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                # Found close — skip span content + tag, continue
                buf = buf[idx + len(self._CLOSE_TAG):]
                self._in_span = False
            else:
                idx = buf.lower().find(self._OPEN_TAG)
                if idx == -1:
                    # No open tag — hold back a potential partial open tag
                    held = self._max_partial_suffix(buf, self._OPEN_TAG)
                    if held:
                        out.append(buf[:-held])
                        self._buf = buf[-held:]
                    else:
                        out.append(buf)
                    return "".join(out)
                # Emit text before the tag, enter span
                if idx > 0:
                    out.append(buf[:idx])
                buf = buf[idx + len(self._OPEN_TAG):]
                self._in_span = True

        return "".join(out)

    def flush(self) -> str:
        """Emit any held-back buffer at end-of-stream.

        If still inside an unterminated span the remaining content is
        discarded (safer: leaking partial memory context is worse than a
        truncated answer). Otherwise the held-back partial-tag tail is
        emitted verbatim (it turned out not to be a real tag).
        """
        if self._in_span:
            self._buf = ""
            self._in_span = False
            return ""
        tail = self._buf
        self._buf = ""
        return tail

    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        """Return the length of the longest buf-suffix that is a tag-prefix.

        Case-insensitive. Returns 0 if no suffix could start the tag.
        """
        tag_lower = tag.lower()
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), len(tag_lower) - 1)
        for i in range(max_check, 0, -1):
            if tag_lower.startswith(buf_lower[-i:]):
                return i
        return 0
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_streaming_memory_scrubber.py -v`
Expected: All 25+ tests PASS

- [ ] **Step 3: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/agents/memory/scrubber.py backend/tests/test_streaming_memory_scrubber.py
git commit -m "feat(memory): add StreamingMemoryScrubber state machine for <memory> tag stripping"
```

---

### Task 3: StreamingMemoryScrubber — Export from memory __init__

**Files:**
- Modify: `packages/harness/deerflow/agents/memory/__init__.py`

- [ ] **Step 1: Add export**

Replace the contents of `packages/harness/deerflow/agents/memory/__init__.py` with:

```python
"""Memory system — Hermes-style Agent-driven curated memory."""

from deerflow.agents.memory.prompt import format_memory_block
from deerflow.agents.memory.scrubber import StreamingMemoryScrubber
from deerflow.agents.memory.security import scan_memory_content
from deerflow.agents.memory.store import MemoryStore
from deerflow.agents.memory.tool import create_memory_tool, memory_tool

__all__ = [
    "MemoryStore",
    "StreamingMemoryScrubber",
    "create_memory_tool",
    "format_memory_block",
    "memory_tool",
    "scan_memory_content",
]
```

- [ ] **Step 2: Run existing memory tests to verify no regressions**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_store.py tests/test_memory_tool.py -v`
Expected: All existing tests PASS

- [ ] **Step 3: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/agents/memory/__init__.py
git commit -m "feat(memory): export StreamingMemoryScrubber from memory package"
```

---

### Task 4: StreamingMemoryScrubber — Integration into worker.py

**Files:**
- Modify: `packages/harness/deerflow/runtime/runs/worker.py`

- [ ] **Step 1: Add import at the top of worker.py**

After line 33 (`from deerflow.runtime.stream_bridge import StreamBridge`), add:

```python
from deerflow.agents.memory.scrubber import StreamingMemoryScrubber
```

- [ ] **Step 2: Create scrubber helper function**

Add this function before the `_lg_mode_to_sse_event` function (before line 509):

```python
def _scrub_chunk(chunk: Any, mode: str, scrubber: StreamingMemoryScrubber) -> Any:
    """Scrub <memory> tags from AIMessageChunk content before SSE publishing."""
    if mode != "messages":
        return chunk
    from langchain_core.messages import AIMessageChunk

    if not isinstance(chunk, tuple) or len(chunk) != 2:
        return chunk
    msg, metadata = chunk
    if not isinstance(msg, AIMessageChunk):
        return chunk
    content = msg.content
    if not isinstance(content, str) or not content:
        return chunk
    scrubbed = scrubber.feed(content)
    if scrubbed == content:
        return chunk
    return (msg.model_copy(update={"content": scrubbed}), metadata)
```

- [ ] **Step 3: Initialize scrubber in run_agent**

After line 269 (`logger.info("Run %s: streaming with modes %s (requested: %s)", run_id, lg_modes, requested_modes)`), add:

```python
        scrubber = StreamingMemoryScrubber()
```

- [ ] **Step 4: Apply scrubber in the single-mode streaming loop**

Change lines 275-280 from:

```python
            async for chunk in agent.astream(graph_input, config=runnable_config, stream_mode=single_mode):
                if record.abort_event.is_set():
                    logger.info("Run %s abort requested — stopping", run_id)
                    break
                sse_event = _lg_mode_to_sse_event(single_mode)
                await bridge.publish(run_id, sse_event, serialize(chunk, mode=single_mode))
```

To:

```python
            async for chunk in agent.astream(graph_input, config=runnable_config, stream_mode=single_mode):
                if record.abort_event.is_set():
                    logger.info("Run %s abort requested — stopping", run_id)
                    break
                chunk = _scrub_chunk(chunk, single_mode, scrubber)
                sse_event = _lg_mode_to_sse_event(single_mode)
                await bridge.publish(run_id, sse_event, serialize(chunk, mode=single_mode))
```

- [ ] **Step 5: Apply scrubber in the multi-mode streaming loop**

Change lines 293-298 from:

```python
                mode, chunk = _unpack_stream_item(item, lg_modes, stream_subgraphs)
                if mode is None:
                    continue

                sse_event = _lg_mode_to_sse_event(mode)
                await bridge.publish(run_id, sse_event, serialize(chunk, mode=mode))
```

To:

```python
                mode, chunk = _unpack_stream_item(item, lg_modes, stream_subgraphs)
                if mode is None:
                    continue

                chunk = _scrub_chunk(chunk, mode, scrubber)
                sse_event = _lg_mode_to_sse_event(mode)
                await bridge.publish(run_id, sse_event, serialize(chunk, mode=mode))
```

- [ ] **Step 6: Flush scrubber in the finally block**

After line 391 (`await bridge.publish_end(run_id)`), add:

```python
        scrubber.flush()
```

- [ ] **Step 7: Run existing tests to verify no regressions**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_stream_bridge.py tests/test_streaming_memory_scrubber.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/runtime/runs/worker.py
git commit -m "feat(memory): integrate StreamingMemoryScrubber into SSE streaming pipeline"
```

---

### Task 5: MemoryConfig — Add nudge_interval field

**Files:**
- Modify: `packages/harness/deerflow/config/memory_config.py`

- [ ] **Step 1: Add nudge_interval field**

Replace the contents of `packages/harness/deerflow/config/memory_config.py` with:

```python
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
```

- [ ] **Step 2: Verify existing config tests pass**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_store.py tests/test_memory_tool.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/config/memory_config.py
git commit -m "feat(memory): add nudge_interval config field for background memory reviews"
```

---

### Task 6: MemoryNudgeMiddleware — Tests

**Files:**
- Create: `tests/test_memory_nudge_middleware.py`

- [ ] **Step 1: Write all nudge middleware tests**

```python
"""Tests for MemoryNudgeMiddleware background memory review."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares.memory_nudge_middleware import (
    MEMORY_REVIEW_PROMPT,
    MAX_REVIEW_MESSAGES,
    MemoryNudgeMiddleware,
)
from deerflow.config.memory_config import MemoryConfig, set_memory_config


def _make_runtime():
    runtime = MagicMock()
    runtime.context = {"thread_id": "test-thread", "run_id": "test-run"}
    return runtime


def _make_state(user_count: int, ai_count: int = 0):
    messages = []
    for i in range(user_count):
        messages.append(HumanMessage(content=f"User message {i}"))
    for i in range(ai_count):
        messages.append(AIMessage(content=f"AI response {i}"))
    return {"messages": messages}


class TestMemoryNudgeMiddlewareTurnCounting:
    def test_no_trigger_below_interval(self):
        set_memory_config(MemoryConfig(enabled=True, nudge_interval=5))
        middleware = MemoryNudgeMiddleware()
        state = _make_state(user_count=3, ai_count=3)
        result = middleware.after_model(state, _make_runtime())
        assert result is None

    def test_no_trigger_when_interval_is_zero(self):
        set_memory_config(MemoryConfig(enabled=True, nudge_interval=0))
        middleware = MemoryNudgeMiddleware()
        state = _make_state(user_count=10, ai_count=10)
        result = middleware.after_model(state, _make_runtime())
        assert result is None

    @patch("deerflow.agents.middlewares.memory_nudge_middleware.MemoryNudgeMiddleware._spawn_review")
    def test_triggers_at_interval(self, mock_spawn):
        set_memory_config(MemoryConfig(enabled=True, nudge_interval=5))
        middleware = MemoryNudgeMiddleware()
        state = _make_state(user_count=5, ai_count=5)
        middleware.after_model(state, _make_runtime())
        mock_spawn.assert_called_once()

    @patch("deerflow.agents.middlewares.memory_nudge_middleware.MemoryNudgeMiddleware._spawn_review")
    def test_triggers_at_double_interval(self, mock_spawn):
        set_memory_config(MemoryConfig(enabled=True, nudge_interval=5))
        middleware = MemoryNudgeMiddleware()
        state = _make_state(user_count=10, ai_count=10)
        middleware.after_model(state, _make_runtime())
        mock_spawn.assert_called_once()

    @patch("deerflow.agents.middlewares.memory_nudge_middleware.MemoryNudgeMiddleware._spawn_review")
    def test_does_not_retrigger_same_count(self, mock_spawn):
        set_memory_config(MemoryConfig(enabled=True, nudge_interval=5))
        middleware = MemoryNudgeMiddleware()
        state = _make_state(user_count=5, ai_count=5)
        middleware.after_model(state, _make_runtime())
        assert mock_spawn.call_count == 1
        # Call again with same count — should not re-trigger
        middleware.after_model(state, _make_runtime())
        assert mock_spawn.call_count == 1

    @patch("deerflow.agents.middlewares.memory_nudge_middleware.MemoryNudgeMiddleware._spawn_review")
    def test_triggers_again_at_next_interval(self, mock_spawn):
        set_memory_config(MemoryConfig(enabled=True, nudge_interval=5))
        middleware = MemoryNudgeMiddleware()
        # First trigger at 5
        middleware.after_model(_make_state(user_count=5, ai_count=5), _make_runtime())
        assert mock_spawn.call_count == 1
        # Second trigger at 10
        middleware.after_model(_make_state(user_count=10, ai_count=10), _make_runtime())
        assert mock_spawn.call_count == 2

    @patch("deerflow.agents.middlewares.memory_nudge_middleware.MemoryNudgeMiddleware._spawn_review")
    def test_no_trigger_between_intervals(self, mock_spawn):
        set_memory_config(MemoryConfig(enabled=True, nudge_interval=10))
        middleware = MemoryNudgeMiddleware()
        state = _make_state(user_count=7, ai_count=7)
        middleware.after_model(state, _make_runtime())
        mock_spawn.assert_not_called()


class TestMemoryNudgeMiddlewareBackgroundReview:
    @pytest.mark.asyncio
    async def test_run_background_review_handles_exception(self, tmp_path):
        set_memory_config(MemoryConfig(enabled=True, nudge_interval=5))
        middleware = MemoryNudgeMiddleware()

        messages = [HumanMessage(content="test"), AIMessage(content="reply")]

        # Should not raise even if everything fails
        with patch(
            "deerflow.agents.middlewares.memory_nudge_middleware.create_chat_model",
            side_effect=RuntimeError("model creation failed"),
        ):
            await middleware._run_background_review(messages, None)

    @pytest.mark.asyncio
    async def test_run_background_review_with_store(self, tmp_path):
        from deerflow.agents.memory.store import MemoryStore

        store = MemoryStore(memory_dir=tmp_path / "mem")
        store.load_from_disk()
        set_memory_config(MemoryConfig(enabled=True, nudge_interval=5))
        middleware = MemoryNudgeMiddleware()

        messages = [HumanMessage(content="I prefer Python"), AIMessage(content="Noted")]

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Nothing to save.")]})

        with patch(
            "deerflow.agents.middlewares.memory_nudge_middleware.create_chat_model",
            return_value=MagicMock(),
        ), patch(
            "deerflow.agents.middlewares.memory_nudge_middleware.create_agent",
            return_value=mock_agent,
        ):
            await middleware._run_background_review(messages, store)
            mock_agent.ainvoke.assert_called_once()

    def test_review_prompt_is_defined(self):
        assert "memory" in MEMORY_REVIEW_PROMPT.lower()
        assert "save" in MEMORY_REVIEW_PROMPT.lower()

    def test_max_review_messages_is_reasonable(self):
        assert MAX_REVIEW_MESSAGES >= 10
        assert MAX_REVIEW_MESSAGES <= 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_nudge_middleware.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deerflow.agents.middlewares.memory_nudge_middleware'`

---

### Task 7: MemoryNudgeMiddleware — Implementation

**Files:**
- Create: `packages/harness/deerflow/agents/middlewares/memory_nudge_middleware.py`

- [ ] **Step 1: Implement the middleware**

```python
"""Memory nudge middleware — periodically triggers background memory review."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, override

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deerflow.config.memory_config import get_memory_config
from deerflow.models import create_chat_model

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool.\n"
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)

MAX_REVIEW_MESSAGES = 20


class MemoryNudgeMiddlewareState(AgentState):
    """Compatible with the ThreadState schema."""

    pass


class MemoryNudgeMiddleware(AgentMiddleware[MemoryNudgeMiddlewareState]):
    """Periodically spawns a background memory review.

    Counts user messages in ``after_model``. When the count reaches a multiple
    of the configured ``nudge_interval``, a background ``asyncio.Task`` is
    spawned to run a lightweight agent that reviews recent conversation history
    and decides whether to save anything to memory.
    """

    state_schema = MemoryNudgeMiddlewareState

    def __init__(self, *, app_config: "AppConfig | None" = None):
        super().__init__()
        self._app_config = app_config
        self._last_triggered_count: int = 0

    @override
    def after_model(self, state: MemoryNudgeMiddlewareState, runtime: Runtime) -> dict | None:
        config = get_memory_config()
        interval = config.nudge_interval
        if interval <= 0:
            return None

        messages = state.get("messages", [])
        user_count = sum(1 for m in messages if getattr(m, "type", None) == "human")

        if user_count <= 0:
            return None

        if user_count % interval != 0:
            return None

        if user_count == self._last_triggered_count:
            return None

        self._last_triggered_count = user_count
        self._spawn_review(messages)
        return None

    def _spawn_review(self, messages: list[Any]) -> None:
        """Spawn a background asyncio task for memory review."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running event loop — skipping memory review")
            return

        from deerflow.agents.memory import MemoryStore, create_memory_tool
        from deerflow.config.paths import get_paths
        from deerflow.runtime.user_context import get_effective_user_id

        try:
            config = get_memory_config()
            paths = get_paths()
            user_id = get_effective_user_id()
            memory_dir = paths.user_dir(user_id) / "memory"
            store = MemoryStore(
                memory_dir=memory_dir,
                memory_char_limit=config.memory_char_limit,
                user_char_limit=config.user_char_limit,
            )
            store.load_from_disk()
        except Exception:
            logger.exception("Failed to create MemoryStore for background review")
            return

        loop.create_task(self._run_background_review(messages, store))

    async def _run_background_review(self, messages: list[Any], store: Any) -> None:
        """Run the background memory review with a lightweight agent."""
        try:
            recent = messages[-MAX_REVIEW_MESSAGES:]
            review_messages = [{"role": "user", "content": MEMORY_REVIEW_PROMPT}]
            for m in recent:
                role = "user" if getattr(m, "type", None) == "human" else "assistant"
                content = m.content if isinstance(m.content, str) else str(m.content)
                review_messages.append({"role": role, "content": content})

            model = create_chat_model(thinking_enabled=False, app_config=self._app_config)
            tools = []
            if store is not None:
                tools.append(create_memory_tool(store))

            agent = create_agent(model=model, tools=tools if tools else None)
            await agent.ainvoke({"messages": review_messages})
            logger.info("Background memory review completed")
        except Exception:
            logger.warning("Background memory review failed", exc_info=True)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_nudge_middleware.py -v`
Expected: All 10 tests PASS

- [ ] **Step 3: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/agents/middlewares/memory_nudge_middleware.py
git commit -m "feat(memory): add MemoryNudgeMiddleware with background review agent"
```

---

### Task 8: MemoryNudgeMiddleware — Integration into middleware chain

**Files:**
- Modify: `packages/harness/deerflow/agents/lead_agent/agent.py`

- [ ] **Step 1: Add import at the top of agent.py**

After line 17 (`from deerflow.agents.middlewares.todo_middleware import TodoMiddleware`), add:

```python
from deerflow.agents.middlewares.memory_nudge_middleware import MemoryNudgeMiddleware
```

- [ ] **Step 2: Add middleware to the chain**

In `_build_middlewares()`, after line 293 (`middlewares.append(LoopDetectionMiddleware())`) and before line 295 (`# Inject custom middlewares before ClarificationMiddleware`), add:

```python
    # MemoryNudgeMiddleware — periodic background memory review
    if resolved_app_config.memory.enabled and resolved_app_config.memory.nudge_interval > 0:
        middlewares.append(MemoryNudgeMiddleware(app_config=resolved_app_config))
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_create_deerflow_agent.py tests/test_memory_nudge_middleware.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
cd /root/deer-flow
git add backend/packages/harness/deerflow/agents/lead_agent/agent.py
git commit -m "feat(memory): register MemoryNudgeMiddleware in lead agent middleware chain"
```

---

### Task 9: Config documentation — Update config.example.yaml

**Files:**
- Modify: `config.example.yaml`

- [ ] **Step 1: Add nudge_interval to memory config section**

Change the memory section (lines 787-792) from:

```yaml
memory:
  enabled: true
  injection_enabled: true
  storage_path: .deer-flow
  memory_char_limit: 2200
  user_char_limit: 1375
```

To:

```yaml
memory:
  enabled: true
  injection_enabled: true
  storage_path: .deer-flow
  memory_char_limit: 2200
  user_char_limit: 1375
  nudge_interval: 10  # User turns between background memory reviews; 0 = disabled
```

- [ ] **Step 2: Commit**

```bash
cd /root/deer-flow
git add config.example.yaml
git commit -m "docs(memory): add nudge_interval to example config"
```

---

### Task 10: Full regression test suite

**Files:**
- No file changes — verification only

- [ ] **Step 1: Run the complete test suite**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/ -v --timeout=60 -x`
Expected: All tests PASS (2900+ tests)

- [ ] **Step 2: Run only memory-related tests for a final check**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_store.py tests/test_memory_tool.py tests/test_streaming_memory_scrubber.py tests/test_memory_nudge_middleware.py tests/test_memory_router.py -v --timeout=30`
Expected: All tests PASS
