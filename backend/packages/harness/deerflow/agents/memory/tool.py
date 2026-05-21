"""Memory tool for Agent-driven memory management — ported from Hermes."""

import json
from typing import Annotated, Any

from langchain.tools import tool
from pydantic import Field

from deerflow.agents.memory.store import MemoryStore


@tool("memory")
def memory_tool(
    action: Annotated[str, Field(description="The action to perform: add, replace, or remove.")],
    target: Annotated[str, Field(description="Which memory store: 'memory' for personal notes, 'user' for user profile.")],
    content: Annotated[str | None, Field(description="The entry content. Required for 'add' and 'replace'.")] = None,
    old_text: Annotated[str | None, Field(description="Short unique substring identifying the entry to replace or remove.")] = None,
    _store: Any = None,
) -> str:
    """Save durable information to persistent memory that survives across sessions.
    Memory is injected into future turns, so keep it compact and focused on facts
    that will still matter later.

    WHEN TO SAVE (do this proactively, don't wait to be asked):
    - User corrects you or says 'remember this' / 'don't do that again'
    - User shares a preference, habit, or personal detail (name, role, timezone, coding style)
    - You discover something about the environment (OS, installed tools, project structure)
    - You learn a convention, API quirk, or workflow specific to this user's setup
    - You identify a stable fact that will be useful again in future sessions

    PRIORITY: User preferences and corrections > environment facts > procedural knowledge.
    The most valuable memory prevents the user from having to repeat themselves.

    Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO
    state to memory.

    TWO TARGETS:
    - 'user': who the user is — name, role, preferences, communication style, pet peeves
    - 'memory': your notes — environment facts, project conventions, tool quirks, lessons learned

    ACTIONS: add (new entry), replace (update existing — old_text identifies it), remove (delete — old_text identifies it).

    SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state.
    """
    if _store is None:
        return json.dumps({"success": False, "error": "Memory is not available."})

    if target not in ("memory", "user"):
        return json.dumps({"success": False, "error": f"Invalid target '{target}'. Use 'memory' or 'user'."})

    if action == "add":
        if not content:
            return json.dumps({"success": False, "error": "Content is required for 'add' action."})
        result = _store.add(target, content)
    elif action == "replace":
        if not old_text:
            return json.dumps({"success": False, "error": "old_text is required for 'replace' action."})
        if not content:
            return json.dumps({"success": False, "error": "content is required for 'replace' action."})
        result = _store.replace(target, old_text, content)
    elif action == "remove":
        if not old_text:
            return json.dumps({"success": False, "error": "old_text is required for 'remove' action."})
        result = _store.remove(target, old_text)
    else:
        return json.dumps({"success": False, "error": f"Unknown action '{action}'. Use: add, replace, remove."})

    return json.dumps(result, ensure_ascii=False)


def create_memory_tool(store: MemoryStore):
    """Create a memory tool instance with the store bound via closure."""

    def _bound_invoke(
        action: Annotated[str, Field(description="The action to perform: add, replace, or remove.")],
        target: Annotated[str, Field(description="Which memory store: 'memory' for personal notes, 'user' for user profile.")],
        content: Annotated[str | None, Field(description="The entry content. Required for 'add' and 'replace'.")] = None,
        old_text: Annotated[str | None, Field(description="Short unique substring identifying the entry to replace or remove.")] = None,
    ) -> str:
        """Save durable information to persistent memory that survives across sessions.
        Memory is injected into future turns, so keep it compact and focused on facts
        that will still matter later.

        WHEN TO SAVE (do this proactively, don't wait to be asked):
        - User corrects you or says 'remember this' / 'don't do that again'
        - User shares a preference, habit, or personal detail (name, role, timezone, coding style)
        - You discover something about the environment (OS, installed tools, project structure)
        - You learn a convention, API quirk, or workflow specific to this user's setup
        - You identify a stable fact that will be useful again in future sessions

        PRIORITY: User preferences and corrections > environment facts > procedural knowledge.
        The most valuable memory prevents the user from having to repeat themselves.

        Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO
        state to memory.

        TWO TARGETS:
        - 'user': who the user is — name, role, preferences, communication style, pet peeves
        - 'memory': your notes — environment facts, project conventions, tool quirks, lessons learned

        ACTIONS: add (new entry), replace (update existing — old_text identifies it), remove (delete — old_text identifies it).

        SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state.
        """
        return memory_tool.func(action=action, target=target, content=content, old_text=old_text, _store=store)

    # Preserve LangChain tool attributes so ToolNode recognises this as a valid tool.
    _bound_invoke.name = memory_tool.name
    _bound_invoke.description = memory_tool.description
    _bound_invoke.args_schema = memory_tool.args_schema
    _bound_invoke.handle_tool_error = True

    return _bound_invoke
