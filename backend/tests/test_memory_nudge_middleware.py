"""Tests for MemoryNudgeMiddleware background memory review."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares.memory_nudge_middleware import (
    MAX_REVIEW_MESSAGES,
    MEMORY_REVIEW_PROMPT,
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

        with (
            patch(
                "deerflow.agents.middlewares.memory_nudge_middleware.create_chat_model",
                return_value=MagicMock(),
            ),
            patch(
                "deerflow.agents.middlewares.memory_nudge_middleware.create_agent",
                return_value=mock_agent,
            ),
        ):
            await middleware._run_background_review(messages, store)
            mock_agent.ainvoke.assert_called_once()

    def test_review_prompt_is_defined(self):
        assert "memory" in MEMORY_REVIEW_PROMPT.lower()
        assert "save" in MEMORY_REVIEW_PROMPT.lower()

    def test_max_review_messages_is_reasonable(self):
        assert MAX_REVIEW_MESSAGES >= 10
        assert MAX_REVIEW_MESSAGES <= 50
