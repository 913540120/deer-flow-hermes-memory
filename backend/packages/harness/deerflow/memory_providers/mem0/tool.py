"""Mem0 tool schemas and handler helpers."""

from __future__ import annotations

MEM0_SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": "Search your Mem0 long-term memory for relevant context from past conversations.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant memories.",
            },
        },
        "required": ["query"],
    },
}

MEM0_ADD_SCHEMA = {
    "name": "mem0_add",
    "description": "Add a specific memory to your Mem0 store for future recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to remember.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata to attach to the memory.",
            },
        },
        "required": ["content"],
    },
}
