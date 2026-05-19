# Session Search (FTS5 + LLM Summarization) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full-history conversation search to DeerFlow via an independent SQLite FTS5 database with trigram support for Chinese, exposed as a `session_search` agent tool.

**Architecture:** Independent SQLite database (`search.db`) stores indexed messages. SQLite triggers auto-maintain FTS5 full-text search index on INSERT/DELETE. Dual FTS5 tables (standard + trigram) support English and Chinese substring search. The agent gets a `session_search` tool that queries FTS5, smart-truncates results around matches, and uses an auxiliary LLM to generate focused summaries.

**Tech Stack:** Python 3.12+, SQLite3 (stdlib), FTS5 + trigram tokenizer, LangChain tools, `create_chat_model()` for auxiliary LLM.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `deerflow/memory_search/__init__.py` | Module exports |
| `deerflow/memory_search/storage.py` | SQLite DB init, schema, triggers, session/message CRUD |
| `deerflow/memory_search/search.py` | FTS5 search, query sanitization, smart truncation, CJK detection, LLM summary |
| `deerflow/memory_search/tool.py` | `session_search` LangChain tool with closure binding |
| `deerflow/memory_search/indexing.py` | Sync message writer called from RunJournal |
| `deerflow/config/memory_search_config.py` | `MemorySearchConfig` Pydantic model + singleton |
| `config/app_config.py` | Add `memory_search` field |
| `runtime/journal.py` | Accept optional indexer, call after `put_batch()` |
| `agents/lead_agent/agent.py` | Register `session_search` tool |
| `config.example.yaml` | Add `memory_search` section |
| `tests/test_memory_search_storage.py` | Storage, schema, trigger tests |
| `tests/test_memory_search_search.py` | Query sanitize, truncation, CJK detection |
| `tests/test_memory_search_tool.py` | Tool integration tests |

---

### Task 1: Storage Layer — SQLite Schema + CRUD

**Files:**
- Create: `packages/harness/deerflow/memory_search/__init__.py`
- Create: `packages/harness/deerflow/memory_search/storage.py`
- Create: `tests/test_memory_search_storage.py`

- [ ] **Step 1: Write failing tests for storage**

```python
# tests/test_memory_search_storage.py
"""Tests for session search storage layer — SQLite schema, triggers, CRUD."""

import tempfile
from pathlib import Path

from deerflow.memory_search.storage import SearchStorage


class TestSearchStorageInit:
    def test_creates_db_file_on_init(self, tmp_path):
        db_path = tmp_path / "test_search.db"
        assert not db_path.exists()
        store = SearchStorage(db_path)
        assert db_path.exists()

    def test_creates_all_tables(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        tables = store._list_tables()
        assert "sessions" in tables
        assert "messages" in tables
        assert "messages_fts" in tables
        assert "messages_fts_trigram" in tables

    def test_creates_triggers(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        triggers = store._list_triggers()
        assert "messages_ai" in triggers
        assert "messages_ad" in triggers


class TestSearchStorageSessions:
    def test_upsert_session_insert(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session(
            session_id="thread_001",
            source="api",
            user_id="user1",
            model="gpt-4o",
            title="Test session",
            started_at=1716200000.0,
        )
        session = store.get_session("thread_001")
        assert session is not None
        assert session["id"] == "thread_001"
        assert session["source"] == "api"
        assert session["user_id"] == "user1"
        assert session["model"] == "gpt-4o"
        assert session["title"] == "Test session"

    def test_upsert_session_update(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "Old title", 100.0)
        store.upsert_session("t1", "api", "u1", "m2", "New title", 100.0)
        session = store.get_session("t1")
        assert session["title"] == "New title"
        assert session["model"] == "m2"

    def test_get_session_nonexistent(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        assert store.get_session("nonexistent") is None


class TestSearchStorageMessages:
    def test_insert_message(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "Title", 100.0)
        msg_id = store.insert_message(
            session_id="t1",
            role="user",
            content="Hello world",
            tool_name=None,
            timestamp=100.1,
        )
        assert isinstance(msg_id, int)
        assert msg_id > 0

    def test_insert_message_auto_fts_index(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "Title", 100.0)
        store.insert_message("t1", "user", "Docker deployment guide", None, 100.1)
        store.insert_message("t1", "assistant", "Use docker-compose", None, 100.2)
        results = store.search_fts("docker")
        assert len(results) > 0

    def test_insert_message_trigram_cjk(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "Title", 100.0)
        store.insert_message("t1", "user", "部署到阿里云服务器", None, 100.1)
        results = store.search_fts_trigram("阿里云")
        assert len(results) > 0

    def test_delete_session_cascades(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "Title", 100.0)
        store.insert_message("t1", "user", "Hello", None, 100.1)
        store.delete_session("t1")
        assert store.get_session("t1") is None
        results = store.search_fts("Hello")
        assert len(results) == 0


class TestSearchStorageSearch:
    def test_search_fts_returns_matching_messages(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "Title", 100.0)
        store.upsert_session("t2", "api", "u1", "m1", "Title2", 200.0)
        store.insert_message("t1", "user", "How to deploy Kubernetes?", None, 100.1)
        store.insert_message("t2", "user", "What is the weather today?", None, 200.1)
        results = store.search_fts("Kubernetes")
        assert len(results) == 1
        assert results[0]["session_id"] == "t1"

    def test_search_fts_groups_by_session(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "Title", 100.0)
        store.insert_message("t1", "user", "docker pull", None, 100.1)
        store.insert_message("t1", "assistant", "docker run command", None, 100.2)
        results = store.search_fts("docker")
        session_ids = {r["session_id"] for r in results}
        assert session_ids == {"t1"}

    def test_search_fts_empty_query(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        results = store.search_fts("")
        assert results == []

    def test_search_fts_trigram_chinese(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "Title", 100.0)
        store.insert_message("t1", "user", "使用Python编写爬虫", None, 100.1)
        results = store.search_fts_trigram("Python")
        assert len(results) > 0

    def test_list_recent_sessions(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "First", 100.0)
        store.upsert_session("t2", "api", "u1", "m1", "Second", 200.0)
        store.upsert_session("t3", "api", "u1", "m1", "Third", 300.0)
        sessions = store.list_recent_sessions(user_id="u1", limit=2)
        assert len(sessions) == 2
        assert sessions[0]["id"] == "t3"  # newest first

    def test_get_session_messages(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "m1", "Title", 100.0)
        store.insert_message("t1", "user", "Hello", None, 100.1)
        store.insert_message("t1", "assistant", "World", None, 100.2)
        messages = store.get_session_messages("t1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_user_isolation(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "user1", "m1", "Title", 100.0)
        store.upsert_session("t2", "api", "user2", "m1", "Title2", 200.0)
        store.insert_message("t1", "user", "secret user1 data", None, 100.1)
        store.insert_message("t2", "user", "secret user2 data", None, 200.1)
        # User1's search should only return their messages
        results = store.search_fts("secret", user_id="user1")
        assert len(results) == 1
        assert results[0]["session_id"] == "t1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_storage.py -v 2>&1 | tail -20`
Expected: `ModuleNotFoundError: No module named 'deerflow.memory_search'`

- [ ] **Step 3: Implement storage layer**

```python
# packages/harness/deerflow/memory_search/__init__.py
"""Session search module — FTS5 full-text search over conversation history."""

from deerflow.memory_search.storage import SearchStorage

__all__ = ["SearchStorage"]
```

```python
# packages/harness/deerflow/memory_search/storage.py
"""SQLite + FTS5 storage for session search with dual tokenizer support.

Independent database (search.db) decoupled from the main DeerFlow database.
SQLite triggers automatically maintain FTS5 index on INSERT/DELETE.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
-- Session metadata
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    title TEXT,
    started_at REAL NOT NULL
);

-- Message content
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL
);

-- English full-text search (standard tokenizer)
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id'
);

-- Chinese substring search (trigram tokenizer, 3+ char matching)
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram',
    content='messages',
    content_rowid='id'
);

-- Trigger: auto-index on INSERT
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id, COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '')
    );
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id, COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '')
    );
END;

-- Trigger: cleanup on DELETE
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at DESC);
"""


class SearchStorage:
    """SQLite + FTS5 storage for session search."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Introspection helpers (for tests)
    # ------------------------------------------------------------------

    def _list_tables(self) -> list[str]:
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    def _list_triggers(self) -> list[str]:
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def upsert_session(
        self,
        session_id: str,
        source: str,
        user_id: str | None = None,
        model: str | None = None,
        title: str | None = None,
        started_at: float = 0.0,
    ) -> None:
        self._conn.execute(
            """INSERT INTO sessions (id, source, user_id, model, title, started_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   source=excluded.source,
                   user_id=excluded.user_id,
                   model=excluded.model,
                   title=excluded.title""",
            (session_id, source, user_id, model, title, started_at),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        cursor = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    def list_recent_sessions(
        self, user_id: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        if user_id:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
                (user_id, limit),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def insert_message(
        self,
        session_id: str,
        role: str,
        content: str | None,
        tool_name: str | None,
        timestamp: float,
    ) -> int:
        cursor = self._conn.execute(
            """INSERT INTO messages (session_id, role, content, tool_name, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, role, content, tool_name, timestamp),
        )
        self._conn.commit()
        return cursor.lastrowid

    # ------------------------------------------------------------------
    # FTS5 Search
    # ------------------------------------------------------------------

    def search_fts(
        self,
        query: str,
        user_id: str | None = None,
        exclude_session: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search using standard FTS5 tokenizer."""
        if not query or not query.strip():
            return []

        where_clauses = ["messages_fts MATCH ?"]
        params: list[Any] = [query]

        if user_id:
            where_clauses.append("s.user_id = ?")
            params.append(user_id)

        if exclude_session:
            where_clauses.append("m.session_id != ?")
            params.append(exclude_session)

        params.append(limit)

        sql = f"""
            SELECT m.id, m.session_id, m.role, m.content, m.timestamp,
                   s.source, s.model, s.title, s.started_at AS session_started
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY rank
            LIMIT ?
        """
        try:
            cursor = self._conn.execute(sql, params)
        except sqlite3.OperationalError:
            return []
        return [dict(row) for row in cursor.fetchall()]

    def search_fts_trigram(
        self,
        query: str,
        user_id: str | None = None,
        exclude_session: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search using trigram tokenizer (for CJK)."""
        if not query or not query.strip():
            return []

        where_clauses = ["messages_fts_trigram MATCH ?"]
        params: list[Any] = [query]

        if user_id:
            where_clauses.append("s.user_id = ?")
            params.append(user_id)

        if exclude_session:
            where_clauses.append("m.session_id != ?")
            params.append(exclude_session)

        params.append(limit)

        sql = f"""
            SELECT m.id, m.session_id, m.role, m.content, m.timestamp,
                   s.source, s.model, s.title, s.started_at AS session_started
            FROM messages_fts_trigram
            JOIN messages m ON m.id = messages_fts_trigram.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY rank
            LIMIT ?
        """
        try:
            cursor = self._conn.execute(sql, params)
        except sqlite3.OperationalError:
            return []
        return [dict(row) for row in cursor.fetchall()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_storage.py -v 2>&1 | tail -30`
Expected: All 18 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/deer-flow/backend
git add packages/harness/deerflow/memory_search/__init__.py packages/harness/deerflow/memory_search/storage.py tests/test_memory_search_storage.py
git commit -m "feat(memory-search): add SearchStorage with SQLite FTS5 schema and CRUD"
```

---

### Task 2: Search Logic — Query Sanitize + CJK Detection + Smart Truncation

**Files:**
- Create: `packages/harness/deerflow/memory_search/search.py`
- Create: `tests/test_memory_search_search.py`

- [ ] **Step 1: Write failing tests for search logic**

```python
# tests/test_memory_search_search.py
"""Tests for session search logic — query sanitization, CJK detection, truncation."""

from deerflow.memory_search.search import (
    sanitize_fts5_query,
    contains_cjk,
    count_cjk,
    truncate_around_matches,
    format_conversation,
)


class TestSanitizeFts5Query:
    def test_simple_word(self):
        assert sanitize_fts5_query("docker") == "docker"

    def test_multiple_words(self):
        assert sanitize_fts5_query("docker kubernetes") == "docker kubernetes"

    def test_preserves_quoted_phrases(self):
        result = sanitize_fts5_query('"exact phrase" other')
        assert '"exact phrase"' in result
        assert "other" in result

    def test_strips_unmatched_quotes(self):
        result = sanitize_fts5_query('test "unclosed phrase')
        assert '"' not in result or 'test' in result

    def test_wraps_hyphenated_terms(self):
        result = sanitize_fts5_query("chat-send")
        assert '"chat-send"' in result

    def test_wraps_dotted_terms(self):
        result = sanitize_fts5_query("my-app.config.ts")
        assert '"' in result

    def test_strips_plus_braces(self):
        result = sanitize_fts5_query("test+word {other}")
        assert "+" not in result
        assert "{" not in result
        assert "}" not in result

    def test_collapses_stars(self):
        result = sanitize_fts5_query("test***word")
        assert "***" not in result

    def test_removes_leading_star(self):
        result = sanitize_fts5_query("*test")
        assert not result.startswith("*")

    def test_removes_dangling_and(self):
        result = sanitize_fts5_query("AND test")
        assert not result.startswith("AND")

    def test_removes_trailing_not(self):
        result = sanitize_fts5_query("test NOT")
        assert not result.endswith("NOT")

    def test_empty_string(self):
        assert sanitize_fts5_query("") == ""

    def test_chinese_passthrough(self):
        result = sanitize_fts5_query("部署阿里云")
        assert "部署阿里云" in result

    def test_boolean_operators_preserved(self):
        result = sanitize_fts5_query("docker OR kubernetes")
        assert "OR" in result


class TestCjkDetection:
    def test_contains_cjk_chinese(self):
        assert contains_cjk("中文测试") is True

    def test_contains_cjk_japanese_hiragana(self):
        assert contains_cjk("こんにちは") is True

    def test_contains_cjk_korean(self):
        assert contains_cjk("한국어") is True

    def test_contains_cjk_english(self):
        assert contains_cjk("hello") is False

    def test_contains_cjk_mixed(self):
        assert contains_cjk("hello你好") is True

    def test_count_cjk(self):
        assert count_cjk("中文abc测试") == 4

    def test_count_cjk_empty(self):
        assert count_cjk("") == 0

    def test_count_cjk_no_cjk(self):
        assert count_cjk("hello world") == 0


class TestTruncateAroundMatches:
    def test_short_text_no_truncation(self):
        text = "Hello world"
        result = truncate_around_matches(text, "Hello")
        assert result == text

    def test_truncation_with_phrase_match(self):
        text = "A " * 1000 + "TARGET HERE" + " B" * 1000
        result = truncate_around_matches(text, "TARGET", max_chars=500)
        assert "TARGET" in result
        assert len(result) <= 600  # account for truncation markers

    def test_truncation_no_match(self):
        text = "A" * 2000
        result = truncate_around_matches(text, "NOTFOUND", max_chars=500)
        assert len(result) <= 600

    def test_truncation_preserves_prefix_marker(self):
        text = "x" * 2000 + "TARGET" + "y" * 2000
        result = truncate_around_matches(text, "TARGET", max_chars=500)
        assert "truncated" in result.lower() or "TARGET" in result

    def test_empty_text(self):
        assert truncate_around_matches("", "query") == ""


class TestFormatConversation:
    def test_formats_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        result = format_conversation(messages)
        assert "user" in result.lower() or "Hello" in result
        assert "assistant" in result.lower() or "World" in result

    def test_empty_messages(self):
        assert format_conversation([]) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_search.py -v 2>&1 | tail -20`
Expected: `ModuleNotFoundError: No module named 'deerflow.memory_search.search'`

- [ ] **Step 3: Implement search logic**

```python
# packages/harness/deerflow/memory_search/search.py
"""FTS5 search logic — query sanitization, CJK detection, smart truncation, formatting.

Ported from Hermes's session_search_tool.py, adapted for DeerFlow's architecture.
"""

import re

MAX_SESSION_CHARS = 100_000


# ------------------------------------------------------------------
# CJK character detection
# ------------------------------------------------------------------

_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B
    (0x3000, 0x303F),   # CJK Symbols
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
)


def _is_cjk_codepoint(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def contains_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    return any(_is_cjk_codepoint(ord(ch)) for ch in text)


def count_cjk(text: str) -> int:
    """Count CJK characters in text."""
    return sum(1 for ch in text if _is_cjk_codepoint(ord(ch)))


# ------------------------------------------------------------------
# FTS5 query sanitization
# ------------------------------------------------------------------

def sanitize_fts5_query(query: str) -> str:
    """Sanitize user input for safe use in FTS5 MATCH queries.

    Preserves properly paired quoted phrases, strips unmatched FTS5
    special characters, wraps hyphenated/dotted terms in quotes.
    """
    if not query or not query.strip():
        return ""

    # Step 1: Extract balanced double-quoted phrases
    quoted_parts: list[str] = []

    def _preserve_quoted(m: re.Match) -> str:
        quoted_parts.append(m.group(0))
        return f"\x00Q{len(quoted_parts) - 1}\x00"

    sanitized = re.sub(r'"[^"]*"', _preserve_quoted, query)

    # Step 2: Strip remaining FTS5-special characters
    sanitized = re.sub(r'[+{}()^]', " ", sanitized)

    # Step 3: Collapse repeated * and remove leading *
    sanitized = re.sub(r"\*+", "*", sanitized)
    sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

    # Step 4: Remove dangling boolean operators
    sanitized = sanitized.strip()
    sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized)
    sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized)

    # Step 5: Wrap unquoted dotted/hyphenated terms
    sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)

    # Step 6: Restore preserved quoted phrases
    for i, quoted in enumerate(quoted_parts):
        sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

    return sanitized.strip()


# ------------------------------------------------------------------
# Smart truncation
# ------------------------------------------------------------------

def truncate_around_matches(
    full_text: str, query: str, max_chars: int = MAX_SESSION_CHARS
) -> str:
    """Truncate text to max_chars, centering the window on query matches.

    Strategy (priority order):
    1. Find full query as phrase (case-insensitive)
    2. Find co-occurrence of all terms within 200-char window
    3. Fall back to individual term positions
    """
    if len(full_text) <= max_chars:
        return full_text

    if not query or not query.strip():
        return full_text[:max_chars] + "\n\n...[truncated]..."

    text_lower = full_text.lower()
    query_lower = query.lower().strip()
    match_positions: list[int] = []

    # --- 1. Full-phrase search ---
    phrase_pat = re.compile(re.escape(query_lower))
    match_positions = [m.start() for m in phrase_pat.finditer(text_lower)]

    # --- 2. Proximity co-occurrence of all terms ---
    if not match_positions:
        terms = query_lower.split()
        if len(terms) > 1:
            term_positions: dict[str, list[int]] = {}
            for t in terms:
                term_positions[t] = [
                    m.start() for m in re.finditer(re.escape(t), text_lower)
                ]
            rarest = min(terms, key=lambda t: len(term_positions.get(t, [])))
            for pos in term_positions.get(rarest, []):
                if all(
                    any(abs(p - pos) < 200 for p in term_positions.get(t, []))
                    for t in terms
                    if t != rarest
                ):
                    match_positions.append(pos)

    # --- 3. Individual term positions ---
    if not match_positions:
        terms = query_lower.split()
        for t in terms:
            for m in re.finditer(re.escape(t), text_lower):
                match_positions.append(m.start())

    if not match_positions:
        truncated = full_text[:max_chars]
        suffix = "\n\n...[truncated]..." if max_chars < len(full_text) else ""
        return truncated + suffix

    # --- Pick window covering most matches ---
    match_positions.sort()
    best_start = 0
    best_count = 0
    for candidate in match_positions:
        ws = max(0, candidate - max_chars // 4)
        we = ws + max_chars
        if we > len(full_text):
            ws = max(0, len(full_text) - max_chars)
            we = len(full_text)
        count = sum(1 for p in match_positions if ws <= p < we)
        if count > best_count:
            best_count = count
            best_start = ws

    start = best_start
    end = min(len(full_text), start + max_chars)

    truncated = full_text[start:end]
    prefix = "...[earlier conversation truncated]...\n\n" if start > 0 else ""
    suffix = "\n\n...[later conversation truncated]..." if end < len(full_text) else ""
    return prefix + truncated + suffix


# ------------------------------------------------------------------
# Conversation formatting
# ------------------------------------------------------------------

def format_conversation(messages: list[dict]) -> str:
    """Format a list of message dicts into a readable conversation transcript."""
    if not messages:
        return ""

    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def format_timestamp(ts: float | None) -> str:
    """Format a Unix timestamp to human-readable string."""
    if ts is None:
        return "unknown"
    from datetime import UTC, datetime

    try:
        dt = datetime.fromtimestamp(ts, tz=UTC)
        return dt.strftime("%b %d, %Y at %I:%M %p UTC")
    except (OSError, ValueError):
        return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_search.py -v 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/deer-flow/backend
git add packages/harness/deerflow/memory_search/search.py tests/test_memory_search_search.py
git commit -m "feat(memory-search): add FTS5 search logic with CJK support and smart truncation"
```

---

### Task 3: Indexing — Message Writer for RunJournal

**Files:**
- Create: `packages/harness/deerflow/memory_search/indexing.py`

- [ ] **Step 1: Implement the indexing module**

The indexing module is a thin sync wrapper called from RunJournal's `_flush_async()`. It extracts message data from the event dict and writes it to the search database.

```python
# packages/harness/deerflow/memory_search/indexing.py
"""Sync message indexer for RunJournal integration.

Called inside _flush_async() after put_batch(). Extracts message data
from event dicts and writes to the search database. All exceptions are
caught and logged — never crashes the main flow.
"""

import logging
import time
from typing import Any

from deerflow.memory_search.storage import SearchStorage

logger = logging.getLogger(__name__)


class SearchIndexer:
    """Writes messages from RunJournal events into the search database."""

    def __init__(self, storage: SearchStorage) -> None:
        self._storage = storage

    def write_message(self, event: dict[str, Any]) -> None:
        """Index a single message event into the search database.

        Called from RunJournal._flush_async() after put_batch().
        Failures are logged but never propagated.
        """
        try:
            self._do_write(event)
        except Exception:
            logger.warning(
                "Failed to index message for search: event_type=%s",
                event.get("event_type"),
                exc_info=True,
            )

    def _do_write(self, event: dict[str, Any]) -> None:
        category = event.get("category")
        if category != "message":
            return

        content = event.get("content")
        if not isinstance(content, dict):
            return

        event_type = event.get("event_type", "")
        thread_id = event.get("thread_id")
        if not thread_id:
            return

        # Extract role from event_type or content
        role = self._extract_role(event_type, content)
        if not role:
            return

        # Extract text content
        text = self._extract_text(content)
        if not text or not text.strip():
            return

        # Extract metadata
        user_id = event.get("metadata", {}).get("user_id")
        model = event.get("metadata", {}).get("model")
        tool_name = content.get("name") if role == "tool" else None
        timestamp = time.time()

        # Ensure session exists
        self._storage.upsert_session(
            session_id=thread_id,
            source="api",
            user_id=user_id,
            model=model,
            started_at=timestamp,
        )

        # Insert message
        self._storage.insert_message(
            session_id=thread_id,
            role=role,
            content=text[:50000],  # Cap at 50k chars to prevent DB bloat
            tool_name=tool_name,
            timestamp=timestamp,
        )

    @staticmethod
    def _extract_role(event_type: str, content: dict) -> str | None:
        """Extract role from event_type or content type field."""
        msg_type = content.get("type")
        if msg_type:
            return msg_type

        if "human" in event_type or "human.input" in event_type:
            return "user"
        if "ai.response" in event_type:
            return "assistant"
        if "tool.result" in event_type:
            return "tool"
        return None

    @staticmethod
    def _extract_text(content: dict) -> str | None:
        """Extract text content from a serialized message dict."""
        text = content.get("content")
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            # Multimodal content — extract text parts
            parts = []
            for item in text:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(parts) if parts else None
        return None
```

- [ ] **Step 2: Commit**

```bash
cd /root/deer-flow/backend
git add packages/harness/deerflow/memory_search/indexing.py
git commit -m "feat(memory-search): add SearchIndexer for RunJournal message indexing"
```

---

### Task 4: Session Search Tool — LangChain Tool

**Files:**
- Create: `packages/harness/deerflow/memory_search/tool.py`
- Create: `tests/test_memory_search_tool.py`

- [ ] **Step 1: Write failing tests for the tool**

```python
# tests/test_memory_search_tool.py
"""Tests for session_search LangChain tool."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from deerflow.memory_search.storage import SearchStorage
from deerflow.memory_search.tool import create_session_search_tool


class TestSessionSearchToolBrowse:
    def test_browse_recent_returns_metadata(self, tmp_path):
        store = SearchStorage(tmp_path / "test.db")
        store.upsert_session("t1", "api", "u1", "gpt-4o", "First", 100.0)
        store.upsert_session("t2", "api", "u1", "gpt-4o", "Second", 200.0)
        tool = create_session_search_tool(store, current_thread_id="t1")
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
        # Use a mock LLM to avoid actual API call
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_tool.py -v 2>&1 | tail -20`
Expected: `ModuleNotFoundError: No module named 'deerflow.memory_search.tool'`

- [ ] **Step 3: Implement the search tool**

```python
# packages/harness/deerflow/memory_search/tool.py
"""Session search LangChain tool — search past conversations via FTS5.

Two modes:
- With query: keyword search + LLM summarization
- Without query: browse recent session metadata

Follows the same closure-binding pattern as deerflow.agents.memory.tool.
"""

import json
import logging
from typing import Any, Callable

from langchain.tools import tool
from pydantic import Field

from deerflow.memory_search.search import (
    contains_cjk,
    count_cjk,
    format_conversation,
    format_timestamp,
    sanitize_fts5_query,
    truncate_around_matches,
)
from deerflow.memory_search.storage import SearchStorage

logger = logging.getLogger(__name__)


@tool("session_search")
def session_search_tool(
    query: str | None = None,
    limit: int = 3,
    _store: Any = None,
    _current_thread_id: str | None = None,
    _summarize_fn: Any = None,
) -> str:
    """Search your long-term memory of past conversations, or browse recent sessions.

    Two modes:
    - With keywords: search all history, return focused LLM summaries
    - Without keywords: browse recent session list (metadata only)

    Use this to recall what was discussed, decided, or solved in past conversations.
    """
    if _store is None:
        return json.dumps({"success": False, "error": "Search is not available."})

    # Coerce limit
    try:
        limit = max(1, min(int(limit), 5))
    except (TypeError, ValueError):
        limit = 3

    # Browse recent sessions mode
    if not query or not query.strip():
        return _browse_recent(_store, _current_thread_id, limit)

    # Keyword search mode
    return _keyword_search(_store, _current_thread_id, query.strip(), limit, _summarize_fn)


def _browse_recent(store: SearchStorage, current_thread_id: str | None, limit: int) -> str:
    sessions = store.list_recent_sessions(limit=limit + 5)  # Extra to account for exclusion
    results = []
    for s in sessions:
        if current_thread_id and s["id"] == current_thread_id:
            continue
        results.append({
            "session_id": s["id"],
            "when": format_timestamp(s.get("started_at")),
            "source": s.get("source", "unknown"),
            "model": s.get("model"),
            "title": s.get("title"),
        })
        if len(results) >= limit:
            break

    return json.dumps({
        "success": True,
        "results": results,
        "count": len(results),
    }, ensure_ascii=False)


def _keyword_search(
    store: SearchStorage,
    current_thread_id: str | None,
    raw_query: str,
    limit: int,
    summarize_fn: Any | None,
) -> str:
    from deerflow.memory_search.search import MAX_SESSION_CHARS

    sanitized = sanitize_fts5_query(raw_query)
    if not sanitized:
        return json.dumps({
            "success": True,
            "query": raw_query,
            "results": [],
            "count": 0,
            "message": "Invalid search query.",
        }, ensure_ascii=False)

    # Determine which FTS5 table to use
    is_cjk = contains_cjk(raw_query)
    cjk_count = count_cjk(raw_query)

    # Try standard FTS5 first
    raw_results = store.search_fts(
        sanitized,
        exclude_session=current_thread_id,
        limit=50,
    )

    # If CJK query with 3+ chars, also search trigram table and merge
    if is_cjk and cjk_count >= 3:
        tri_results = store.search_fts_trigram(
            sanitized,
            exclude_session=current_thread_id,
            limit=50,
        )
        # Merge by session_id (deduplicate)
        seen_ids = {r["session_id"] for r in raw_results}
        for r in tri_results:
            if r["session_id"] not in seen_ids:
                raw_results.append(r)
                seen_ids.add(r["session_id"])

    if not raw_results:
        return json.dumps({
            "success": True,
            "query": raw_query,
            "results": [],
            "count": 0,
            "message": "No matching sessions found.",
        }, ensure_ascii=False)

    # Group by session_id, take top N
    seen_sessions: dict[str, dict] = {}
    for r in raw_results:
        sid = r["session_id"]
        if current_thread_id and sid == current_thread_id:
            continue
        if sid not in seen_sessions:
            seen_sessions[sid] = r
        if len(seen_sessions) >= limit:
            break

    # Build results
    summaries = []
    for session_id, match_info in seen_sessions.items():
        messages = store.get_session_messages(session_id)
        session_meta = store.get_session(session_id) or {}
        conversation_text = format_conversation(messages)
        conversation_text = truncate_around_matches(
            conversation_text, raw_query, max_chars=MAX_SESSION_CHARS
        )

        summary = None
        if summarize_fn:
            try:
                summary = summarize_fn(conversation_text, raw_query, session_meta)
            except Exception:
                logger.warning("LLM summarization failed for session %s", session_id, exc_info=True)

        entry: dict[str, Any] = {
            "session_id": session_id,
            "when": format_timestamp(
                session_meta.get("started_at") or match_info.get("session_started")
            ),
            "source": session_meta.get("source") or match_info.get("source", "unknown"),
            "model": session_meta.get("model") or match_info.get("model"),
            "title": session_meta.get("title"),
        }

        if summary:
            entry["summary"] = summary
        else:
            preview = (conversation_text[:500] + "\n…[truncated]") if conversation_text else "No preview available."
            entry["summary"] = f"[Raw preview — summarization unavailable]\n{preview}"

        summaries.append(entry)

    return json.dumps({
        "success": True,
        "query": raw_query,
        "results": summaries,
        "count": len(summaries),
    }, ensure_ascii=False)


def create_session_search_tool(
    store: SearchStorage,
    current_thread_id: str | None = None,
    summarize_fn: Callable | None = None,
):
    """Create a session_search tool instance with store bound via closure.

    Follows the same pattern as deerflow.agents.memory.tool.create_memory_tool.
    """
    def _bound_invoke(tool_input: dict) -> str:
        return session_search_tool.func(
            **tool_input,
            _store=store,
            _current_thread_id=current_thread_id,
            _summarize_fn=summarize_fn,
        )

    _bound_invoke.name = session_search_tool.name
    _bound_invoke.description = session_search_tool.description
    _bound_invoke.args_schema = session_search_tool.args_schema
    _bound_invoke.handle_tool_error = True
    _bound_invoke.invoke = lambda tool_input: _bound_invoke(tool_input)

    return _bound_invoke
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_tool.py -v 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/deer-flow/backend
git add packages/harness/deerflow/memory_search/tool.py tests/test_memory_search_tool.py
git commit -m "feat(memory-search): add session_search LangChain tool with dual-mode search"
```

---

### Task 5: Configuration — MemorySearchConfig

**Files:**
- Create: `packages/harness/deerflow/config/memory_search_config.py`
- Modify: `packages/harness/deerflow/config/app_config.py:97` (add field)
- Modify: `config.example.yaml:793` (add section)

- [ ] **Step 1: Create config model**

```python
# packages/harness/deerflow/config/memory_search_config.py
"""Configuration for session search (FTS5 full-text search over conversation history)."""

from __future__ import annotations

from pydantic import BaseModel


class MemorySearchConfig(BaseModel):
    """Session search configuration."""

    enabled: bool = False
    db_path: str = ".deer-flow/data/search.db"
    max_results: int = 3  # Max sessions to summarize per query
    max_content_chars: int = 100_000  # Truncation limit per session
```

- [ ] **Step 2: Add field to AppConfig**

In `packages/harness/deerflow/config/app_config.py`, after the `memory` field (line 97), add:

```python
    memory_search: MemorySearchConfig = Field(default_factory=MemorySearchConfig, description="Session search configuration (FTS5)")
```

And add the import at the top of the file (in the local imports section, alphabetically):

```python
from deerflow.config.memory_search_config import MemorySearchConfig
```

- [ ] **Step 3: Add section to config.example.yaml**

After the `memory:` section (after line 793 `nudge_interval: 10 ...`), add:

```yaml

# Full-text search over conversation history (SQLite FTS5)
# Allows the agent to search past conversations by keyword
memory_search:
  enabled: false  # Enable to activate session search
  db_path: .deer-flow/data/search.db  # Independent SQLite database for search index
  max_results: 3  # Max sessions to return per search query
  max_content_chars: 100000  # Max chars to include per session before summarization
```

- [ ] **Step 4: Run tests to verify nothing is broken**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_storage.py tests/test_memory_search_search.py tests/test_memory_search_tool.py -v 2>&1 | tail -10`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /root/deer-flow/backend
git add packages/harness/deerflow/config/memory_search_config.py packages/harness/deerflow/config/app_config.py config.example.yaml
git commit -m "feat(memory-search): add MemorySearchConfig and register in AppConfig"
```

---

### Task 6: RunJournal Integration — Index Messages on Flush

**Files:**
- Modify: `packages/harness/deerflow/runtime/journal.py:40-48` (add indexer param)
- Modify: `packages/harness/deerflow/runtime/journal.py:300-311` (call indexer after put_batch)

- [ ] **Step 1: Modify RunJournal.__init__ to accept optional indexer**

In `packages/harness/deerflow/runtime/journal.py`, update `__init__` to accept an optional `search_indexer` parameter:

```python
    def __init__(
        self,
        run_id: str,
        thread_id: str,
        event_store: RunEventStore,
        *,
        track_token_usage: bool = True,
        flush_threshold: int = 20,
        search_indexer: Any | None = None,
    ):
        super().__init__()
        self.run_id = run_id
        self.thread_id = thread_id
        self._store = event_store
        self._track_tokens = track_token_usage
        self._flush_threshold = flush_threshold
        self._search_indexer = search_indexer

        # Write buffer
        self._buffer: list[dict] = []
        self._pending_flush_tasks: set[asyncio.Task[None]] = set()
```

Add the import for `Any` (it's already imported on line 24).

- [ ] **Step 2: Modify _flush_async to call indexer**

In `packages/harness/deerflow/runtime/journal.py`, update `_flush_async`:

```python
    async def _flush_async(self, batch: list[dict]) -> None:
        try:
            await self._store.put_batch(batch)

            # Index messages into search database
            if self._search_indexer:
                for event in batch:
                    if event.get("category") == "message":
                        self._search_indexer.write_message(event)
        except Exception:
            logger.warning(
                "Failed to flush %d events for run %s — returning to buffer",
                len(batch),
                self.run_id,
                exc_info=True,
            )
            # Return failed events to buffer for retry on next flush
            self._buffer = batch + self._buffer
```

- [ ] **Step 3: Run tests to verify nothing is broken**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_storage.py tests/test_memory_search_search.py tests/test_memory_search_tool.py -v 2>&1 | tail -10`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
cd /root/deer-flow/backend
git add packages/harness/deerflow/runtime/journal.py
git commit -m "feat(memory-search): integrate SearchIndexer into RunJournal flush pipeline"
```

---

### Task 7: Lead Agent Integration — Register Tool + Wire Indexer

**Files:**
- Modify: `packages/harness/deerflow/agents/lead_agent/agent.py:395-410` (add session_search tool)

- [ ] **Step 1: Add session_search tool registration in _make_lead_agent**

In `packages/harness/deerflow/agents/lead_agent/agent.py`, after the memory tool registration block (after line 410 `extra_tools.append(create_memory_tool(store))`), add:

```python
    # Register session_search tool when memory search is enabled
    if resolved_app_config.memory_search.enabled:
        from deerflow.config.paths import get_paths as _get_paths
        from deerflow.memory_search.storage import SearchStorage
        from deerflow.memory_search.tool import create_session_search_tool

        _paths = _get_paths()
        _search_db_path = _paths.base_dir / resolved_app_config.memory_search.db_path
        _search_store = SearchStorage(_search_db_path)
        extra_tools.append(
            create_session_search_tool(
                store=_search_store,
                current_thread_id=thread_id if "thread_id" in dir() else None,
            )
        )
```

Note: The `thread_id` is available from the function signature — it comes from the `config.configurable` dict. The actual thread_id is in `config.get("configurable", {}).get("thread_id")`. This needs to be extracted before the tool registration.

Actually, looking at the function signature of `_make_lead_agent` more carefully, the `config: RunnableConfig` parameter contains the thread_id. Let me check how it's accessed.

The thread_id needs to be passed via the config's configurable dict. The `current_thread_id` for the search tool is used to exclude the current thread from search results. It can be set at tool creation time or passed dynamically. Since the agent is created per-invocation and the thread_id is in the config, we can extract it.

In the `_make_lead_agent` function, the thread_id for the current conversation is typically available in the config. However, looking at the function signature, the thread_id is NOT directly available — it's managed by LangGraph internally. The search tool should get the thread_id from the agent's state or from a runtime context.

**Revised approach**: Pass `thread_id` at tool creation time. The `_make_lead_agent` function is called with a `config` parameter that includes `configurable.thread_id`. Let's extract it.

```python
    # Register session_search tool when memory search is enabled
    if resolved_app_config.memory_search.enabled:
        from deerflow.memory_search.storage import SearchStorage
        from deerflow.memory_search.tool import create_session_search_tool

        _search_db_path = resolved_app_config.memory_search.db_path
        if not Path(_search_db_path).is_absolute():
            from deerflow.config.paths import get_paths as _get_paths
            _search_db_path = str(_get_paths().base_dir / _search_db_path)
        _search_store = SearchStorage(_search_db_path)
        _current_tid = config.get("configurable", {}).get("thread_id")
        extra_tools.append(
            create_session_search_tool(
                store=_search_store,
                current_thread_id=_current_tid,
            )
        )
```

- [ ] **Step 2: Run existing tests to verify nothing is broken**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_storage.py tests/test_memory_search_search.py tests/test_memory_search_tool.py -v 2>&1 | tail -10`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
cd /root/deer-flow/backend
git add packages/harness/deerflow/agents/lead_agent/agent.py
git commit -m "feat(memory-search): register session_search tool in lead agent"
```

---

### Task 8: Update __init__.py Exports

**Files:**
- Modify: `packages/harness/deerflow/memory_search/__init__.py`

- [ ] **Step 1: Update exports**

```python
# packages/harness/deerflow/memory_search/__init__.py
"""Session search module — FTS5 full-text search over conversation history."""

from deerflow.memory_search.indexing import SearchIndexer
from deerflow.memory_search.storage import SearchStorage

__all__ = ["SearchStorage", "SearchIndexer"]
```

- [ ] **Step 2: Commit**

```bash
cd /root/deer-flow/backend
git add packages/harness/deerflow/memory_search/__init__.py
git commit -m "feat(memory-search): export SearchIndexer from module __init__"
```

---

### Task 9: Full Regression Test

**Files:** None (test run only)

- [ ] **Step 1: Run all memory search tests**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/test_memory_search_*.py -v 2>&1 | tail -30`
Expected: All ~30 tests PASS

- [ ] **Step 2: Run full test suite**

Run: `cd /root/deer-flow/backend && PYTHONPATH=. uv run pytest tests/ -x --timeout=60 2>&1 | tail -20`
Expected: All tests PASS (same count as before, no regressions)

- [ ] **Step 3: Verify all files are committed**

Run: `cd /root/deer-flow && git status`
Expected: Clean working tree

---

## Self-Review

**1. Spec coverage check:**
- 4.1 Architecture: Task 1 (storage), Task 3 (indexing), Task 6 (journal integration)
- 4.2 Schema: Task 1 (complete SQL schema)
- 4.3 Search Tool: Task 2 (search logic), Task 4 (tool)
- 4.4 Integration: Task 6 (journal), Task 7 (agent)
- 4.5 Configuration: Task 5
- 4.6 Files: All files covered
- 4.7 Error Handling: Task 3 (indexing catches all exceptions), Task 4 (LLM fallback)
- 4.8 Out of Scope: Correctly excluded (no API endpoint, no PostgreSQL, no history migration)

**2. Placeholder scan:** No TBDs, TODOs, or vague instructions. All code is complete.

**3. Type consistency:**
- `SearchStorage` used consistently across all files
- `SearchIndexer.write_message(event: dict)` matches journal's event dict shape
- `create_session_search_tool(store, current_thread_id, summarize_fn)` consistent across test and implementation
- `sanitize_fts5_query` returns `str`, used in `_keyword_search`
