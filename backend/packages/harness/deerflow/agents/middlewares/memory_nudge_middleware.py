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


class MemoryNudgeMiddleware(AgentMiddleware):
    """Periodically spawns a background memory review.

    Counts user messages in ``after_model``. When the count reaches a multiple
    of the configured ``nudge_interval``, a background ``asyncio.Task`` is
    spawned to run a lightweight agent that reviews recent conversation history
    and decides whether to save anything to memory.
    """

    def __init__(self, *, app_config: "AppConfig | None" = None):
        super().__init__()
        self._app_config = app_config
        self._last_triggered_count: int = 0

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
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

        from deerflow.agents.memory import MemoryStore
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
            from deerflow.agents.memory import create_memory_tool

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
