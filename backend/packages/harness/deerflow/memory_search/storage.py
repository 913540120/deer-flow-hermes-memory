"""SQLite + FTS5 storage for session search with dual tokenizer support."""

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

    def _list_tables(self) -> list[str]:
        cursor = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

    def _list_triggers(self) -> list[str]:
        cursor = self._conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

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
        cursor = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    def list_recent_sessions(self, user_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
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

    def search_fts(
        self,
        query: str,
        user_id: str | None = None,
        exclude_session: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
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
            WHERE {" AND ".join(where_clauses)}
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
            WHERE {" AND ".join(where_clauses)}
            ORDER BY rank
            LIMIT ?
        """
        try:
            cursor = self._conn.execute(sql, params)
        except sqlite3.OperationalError:
            return []
        return [dict(row) for row in cursor.fetchall()]
