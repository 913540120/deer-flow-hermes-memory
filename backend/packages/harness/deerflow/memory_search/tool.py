"""Session search LangChain tool — browse recent sessions or keyword search with FTS5."""

import json
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from deerflow.memory_search.search import (
    MAX_SESSION_CHARS,
    contains_cjk,
    count_cjk,
    format_conversation,
    format_timestamp,
    sanitize_fts5_query,
    truncate_around_matches,
)
from deerflow.memory_search.storage import SearchStorage


class SessionSearchInput(BaseModel):
    """Input schema for the session_search tool."""

    query: str | None = Field(default=None, description="Search query. Omit to browse recent sessions.")
    limit: int = Field(default=5, description="Maximum number of results to return.")


@tool("session_search", args_schema=SessionSearchInput)
def session_search_tool(
    query: str | None = None,
    limit: int = 5,
    _store: Any = None,
    _current_thread_id: str | None = None,
    _summarize_fn: Callable | None = None,
) -> str:
    """Search past conversation sessions by keyword, or browse recent sessions.

    Use this tool to find information from previous conversations. Pass a query
    to search by keyword, or omit query to browse recent sessions.
    """
    if _store is None:
        return json.dumps({"success": False, "error": "Session search is not available."})

    try:
        if not query or not query.strip():
            return _browse_recent(_store, _current_thread_id, limit)
        return _keyword_search(_store, _current_thread_id, query, limit, _summarize_fn)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Search failed: {e}"})


def _browse_recent(
    store: SearchStorage,
    current_thread_id: str | None,
    limit: int,
) -> str:
    """Return session metadata for the most recent sessions."""
    sessions = store.list_recent_sessions(limit=limit)
    results = []
    for s in sessions:
        if current_thread_id and s.get("id") == current_thread_id:
            continue
        results.append(
            {
                "session_id": s.get("id"),
                "title": s.get("title"),
                "model": s.get("model"),
                "started_at": format_timestamp(s.get("started_at")),
            }
        )
    return json.dumps({"success": True, "results": results})


def _keyword_search(
    store: SearchStorage,
    current_thread_id: str | None,
    query: str,
    limit: int,
    summarize_fn: Callable | None,
) -> str:
    """FTS5 keyword search with optional CJK trigram merge and LLM summarization."""
    sanitized = sanitize_fts5_query(query)
    if not sanitized:
        return json.dumps({"success": True, "results": [], "count": 0})

    # Search standard FTS table
    rows = store.search_fts(sanitized, exclude_session=current_thread_id, limit=limit * 3)

    # If CJK detected, also search trigram table and merge
    if contains_cjk(query) and count_cjk(query) >= 3:
        tri_rows = store.search_fts_trigram(query, exclude_session=current_thread_id, limit=limit * 3)
        seen_ids = {r["id"] for r in rows}
        for r in tri_rows:
            if r["id"] not in seen_ids:
                rows.append(r)
                seen_ids.add(r["id"])

    # Group by session, preserving insertion order
    sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        sid = row["session_id"]
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "title": row.get("title"),
                "model": row.get("model"),
                "started_at": format_timestamp(row.get("session_started")),
                "matching_messages": [],
            }
        sessions[sid]["matching_messages"].append(
            {
                "role": row.get("role"),
                "content": row.get("content"),
                "timestamp": format_timestamp(row.get("timestamp")),
            }
        )

    # Build results with optional summarization
    results = []
    for sid, meta in sessions.items():
        if len(results) >= limit:
            break

        # Fetch full conversation for the session
        all_messages = store.get_session_messages(sid)
        conversation_text = format_conversation(all_messages)
        truncated = truncate_around_matches(conversation_text, query, MAX_SESSION_CHARS)

        summary = None
        if summarize_fn:
            try:
                summary = summarize_fn(truncated)
            except Exception:
                summary = None

        entry: dict[str, Any] = {
            "session_id": sid,
            "title": meta["title"],
            "model": meta["model"],
            "started_at": meta["started_at"],
        }
        if summary:
            entry["summary"] = summary
        results.append(entry)

    return json.dumps({"success": True, "results": results, "count": len(results)})


def create_session_search_tool(
    store: SearchStorage,
    current_thread_id: str | None = None,
    summarize_fn: Callable | None = None,
):
    """Create a session_search tool instance with store bound via closure."""

    def _bound_invoke(tool_input: dict) -> str:
        return session_search_tool.func(
            **tool_input,
            _store=store,
            _current_thread_id=current_thread_id,
            _summarize_fn=summarize_fn,
        )

    # Preserve LangChain tool attributes
    _bound_invoke.name = session_search_tool.name
    _bound_invoke.description = session_search_tool.description
    _bound_invoke.args_schema = session_search_tool.args_schema
    _bound_invoke.handle_tool_error = True
    _bound_invoke.invoke = lambda tool_input: _bound_invoke(tool_input)

    return _bound_invoke
