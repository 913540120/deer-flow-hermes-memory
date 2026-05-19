"""Tests for session_search LangChain tool."""

import json
from unittest.mock import MagicMock

from deerflow.memory_search.storage import SearchStorage
from deerflow.memory_search.tool import create_session_search_tool


class TestSessionSearchToolBrowse:
    def test_browse_recent_returns_metadata(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "gpt-4o", "First", 100.0)
        store.upsert_session("t2", "api", "u1", "gpt-4o", "Second", 200.0)
        tool = create_session_search_tool(store)
        result = tool({"limit": 5})
        data = json.loads(result)
        assert data["success"] is True
        assert len(data["results"]) == 2

    def test_browse_excludes_current_session(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "gpt-4o", "Current", 100.0)
        store.upsert_session("t2", "api", "u1", "gpt-4o", "Other", 200.0)
        tool = create_session_search_tool(store, current_thread_id="t1")
        result = tool({"limit": 5})
        data = json.loads(result)
        session_ids = [r["session_id"] for r in data["results"]]
        assert "t1" not in session_ids

    def test_browse_empty(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        tool = create_session_search_tool(store)
        result = tool({"limit": 5})
        data = json.loads(result)
        assert data["success"] is True
        assert data["results"] == []


class TestSessionSearchToolKeyword:
    def test_keyword_search_returns_results(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "gpt-4o", "Docker", 100.0)
        store.insert_message("t1", "user", "How to deploy Docker?", None, 100.1)
        store.insert_message("t1", "assistant", "Use docker-compose", None, 100.2)
        mock_llm = MagicMock(return_value="Summary about Docker deployment.")
        tool = create_session_search_tool(store, summarize_fn=mock_llm)
        result = tool({"query": "Docker", "limit": 3})
        data = json.loads(result)
        assert data["success"] is True
        assert len(data["results"]) >= 1
        assert any("Docker" in r.get("summary", "") for r in data["results"])

    def test_keyword_search_no_match(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "gpt-4o", "Title", 100.0)
        store.insert_message("t1", "user", "Hello world", None, 100.1)
        tool = create_session_search_tool(store)
        result = tool({"query": "nonexistent_xyz_12345", "limit": 3})
        data = json.loads(result)
        assert data["success"] is True
        assert data["count"] == 0

    def test_keyword_search_excludes_current_session(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "gpt-4o", "Current", 100.0)
        store.upsert_session("t2", "api", "u1", "gpt-4o", "Other", 200.0)
        store.insert_message("t1", "user", "Docker current session", None, 100.1)
        store.insert_message("t2", "user", "Docker other session", None, 200.1)
        tool = create_session_search_tool(store, current_thread_id="t1")
        result = tool({"query": "Docker", "limit": 3})
        data = json.loads(result)
        session_ids = [r["session_id"] for r in data["results"]]
        assert "t1" not in session_ids


class TestSessionSearchToolAttributes:
    def test_tool_has_name(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        tool = create_session_search_tool(store)
        assert tool.name == "session_search"

    def test_tool_has_description(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        tool = create_session_search_tool(store)
        assert tool.description is not None
        assert len(tool.description) > 0

    def test_tool_has_args_schema(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        tool = create_session_search_tool(store)
        assert tool.args_schema is not None
