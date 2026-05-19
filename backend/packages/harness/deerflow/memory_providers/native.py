"""NativeMemoryProvider — wraps existing MemoryStore as a MemoryProvider."""

from __future__ import annotations

import json

from deerflow.agents.memory.store import MemoryStore
from deerflow.memory_providers.provider import MemoryProvider

_MEMORY_TOOL_SCHEMA = {
    "name": "memory",
    "description": ("Save durable information to persistent memory that survives across sessions."),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action to perform: add, replace, or remove.",
            },
            "target": {
                "type": "string",
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile.",
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'.",
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove.",
            },
        },
        "required": ["action", "target"],
    },
}


class NativeMemoryProvider(MemoryProvider):
    """Wraps MemoryStore as a MemoryProvider for use with MemoryManager."""

    def __init__(self, store: MemoryStore):
        self._store = store

    @property
    def name(self) -> str:
        return "native"

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:

        # Render from live entries rather than the frozen snapshot so that
        # writes made during the current session are visible.
        parts = []
        for target in ("memory", "user"):
            entries = self._store.get_live_entries(target)
            if entries:
                block = self._store._render_block(target, entries)
                parts.append(block)

        if not parts:
            return ""

        content = "\n\n".join(parts)
        return f"<memory>\n{content}\n</memory>"

    def get_tool_schemas(self) -> list[dict]:
        return [_MEMORY_TOOL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name != "memory":
            raise ValueError(f"Unknown tool: {tool_name}")

        target = args.get("target", "memory")
        action = args.get("action", "")
        content = args.get("content")
        old_text = args.get("old_text")

        if target not in ("memory", "user"):
            return json.dumps({"success": False, "error": f"Invalid target '{target}'."})

        if action == "add":
            if not content:
                return json.dumps({"success": False, "error": "Content is required for 'add'."})
            result = self._store.add(target, content)
        elif action == "replace":
            if not old_text:
                return json.dumps({"success": False, "error": "old_text is required for 'replace'."})
            if not content:
                return json.dumps({"success": False, "error": "content is required for 'replace'."})
            result = self._store.replace(target, old_text, content)
        elif action == "remove":
            if not old_text:
                return json.dumps({"success": False, "error": "old_text is required for 'remove'."})
            result = self._store.remove(target, old_text)
        else:
            return json.dumps({"success": False, "error": f"Unknown action '{action}'."})

        return json.dumps(result, ensure_ascii=False)
