# Phase 4 Design: Session Search (FTS5 + LLM Summarization)

Hermes-style session search — Phase 4 of the DeerFlow memory architecture alignment.

## Overview

Phase 4 adds full-history conversation search to DeerFlow, ported from Hermes's
`session_search` tool. The agent can search past conversations using keyword queries
and receive focused LLM-generated summaries, or browse recent session metadata.

The implementation uses an independent SQLite database with FTS5 full-text search
and trigram tokenization for Chinese substring matching. SQLite triggers automatically
maintain the search index on every message INSERT/DELETE.

---

## 4.1 Architecture

```
User message → RunJournal callback → _flush_async()
                                          ↓
                                   put_batch() → run_events table (main DB)
                                          ↓
                              indexing.write_message() → search.db
                                          ↓
                              SQLite triggers → FTS5 index auto-updated

Agent calls session_search tool:
        ↓
search.search_sessions(query)
        ↓
FTS5 MATCH (standard + trigram tables)
        ↓
Group by session_id, take top N, exclude current session
        ↓
_truncate_around_matches() — smart extraction ~100k chars
        ↓
LLM summarization (auxiliary model, temp=0.1, max_tokens=10000)
        ↓
Return focused summaries + metadata
```

### Key Design Decisions

1. **Independent search.db**: Decoupled from main database. SQLite-only, no
   PostgreSQL support needed. Simpler to reason about and faithful to Hermes.
2. **Dual FTS5 tables**: Standard tokenizer for English, trigram tokenizer for
   Chinese substring matching (3+ character matching).
3. **Trigger-based indexing**: SQLite triggers auto-maintain FTS5 index on every
   INSERT/DELETE. Zero application-code overhead for index maintenance.
4. **Sync SQLite write in async journal**: `indexing.write_message()` performs
   a synchronous SQLite INSERT inside `_flush_async()` after `put_batch()`.
   This is safe because SQLite with WAL mode doesn't block readers, and the
   write is fast (no network). Failures are logged but don't affect the main
   event store.
5. **Two search modes**: Keyword search (FTS5 + LLM summary) and recent-session
   browsing (metadata only, no LLM call).
6. **Smart truncation**: `_truncate_around_matches()` extracts ~100k chars
   centered on match positions before sending to LLM.

---

## 4.2 Search Database Schema

```sql
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
    session_id TEXT NOT NULL REFERENCES sessions(id),
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
```

Note: DeerFlow messages are append-only (no UPDATE), so only INSERT and DELETE
triggers are needed — simpler than Hermes which also has UPDATE triggers.

---

## 4.3 Search Tool

### Tool Interface

```python
@tool("session_search")
def session_search_tool(
    query: str | None = None,
    limit: int = 3,
) -> str:
    """Search your long-term memory of past conversations, or browse recent sessions.

    Two modes:
    - With keywords: search all history, return focused LLM summaries
    - Without keywords: browse recent session list (metadata only)

    Use this to recall what was discussed, decided, or solved in past conversations.
    """
```

### Search Modes

**Keyword search** (query provided):
1. Sanitize query — handle FTS5 special characters, preserve quoted phrases,
   support AND/OR/NOT operators
2. Execute dual-table FTS5 search (standard + trigram, take union)
3. Group results by `session_id`, take top N (default 3, max 5)
4. For each session:
   a. Exclude current session and its lineage
   b. Load session messages
   c. `_truncate_around_matches()` — extract ~100k chars centered on matches
   d. LLM summarization with focused prompt (temperature 0.1, max_tokens 10000)
5. Return summaries with metadata

**Browse recent sessions** (query=None):
1. Query recent N sessions ordered by `started_at DESC`
2. Return metadata only (title, time, model, source)
3. No LLM call — instant response

### Query Sanitization

Ported from Hermes's `_sanitize_fts5_query()`:
- Preserve properly paired quoted phrases (`"exact phrase"`)
- Strip unmatched FTS5 special characters
- Wrap hyphenated/dotted terms in quotes
- Support AND, OR, NOT boolean operators
- For queries with 3+ CJK characters, also search the trigram table

### Smart Truncation

`_truncate_around_matches()` strategy (priority order):
1. Find the full query as a phrase (case-insensitive)
2. Find positions where all query terms co-occur within a 200-char window
3. Fall back to individual term positions

Select the window start covering the most candidates, extract up to 100k chars.

### LLM Summary Prompt

```
You are reviewing a past conversation transcript to help recall what happened.
Summarize the conversation with a focus on the search topic. Include:
1. What the user asked about or wanted to accomplish
2. What actions were taken and what the outcomes were
3. Key decisions, solutions found, or conclusions reached
4. Any specific commands, files, URLs, or technical details
5. Anything left unresolved or notable
```

### Response Format

```json
{
    "success": true,
    "query": "docker networking",
    "results": [
        {
            "session_id": "thread_abc123",
            "when": "May 11, 2026 at 11:34 PM",
            "source": "api",
            "model": "claude-sonnet-4-20250514",
            "title": "Docker deployment help",
            "summary": "The user asked about Docker networking issues..."
        }
    ],
    "count": 1
}
```

---

## 4.4 Integration

### Indexing — Message Write Hook

In `runtime/journal.py`, `_flush_async()` method, after successful `put_batch()`:

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
        logger.warning(...)
```

The `RunJournal.__init__` receives an optional `search_indexer` parameter. If
`memory_search.enabled` is True, the indexer is created and passed in; otherwise
it is None and indexing is skipped.

The `indexing.write_message()` function:
- Extracts role, content, thread_id from the event dict
- Resolves user_id via `get_effective_user_id()`
- Ensures the session exists in the `sessions` table (INSERT OR IGNORE)
- INSERTs the message into the `messages` table
- Triggers auto-update the FTS5 index
- All failures caught and logged — never crashes the main flow

### Tool Registration

In `agents/lead_agent/agent.py`, in the tool registration section:

```python
if resolved_app_config.memory.enabled and resolved_app_config.memory_search.enabled:
    tools.append(create_session_search_tool(search_db_path))
```

### Current Session Exclusion

The search tool needs to know the current thread_id to exclude it from results.
This is passed when creating the tool instance, read from the agent's state.

---

## 4.5 Configuration

New file: `config/memory_search_config.py`

```python
class MemorySearchConfig(BaseModel):
    enabled: bool = True
    db_path: str = ".deer-flow/data/search.db"
    max_results: int = 3           # Max sessions to summarize per query
    max_content_chars: int = 100000  # Truncation limit per session
```

In `config/app_config.py`, add:
```python
memory_search: MemorySearchConfig = MemorySearchConfig()
```

In `config.example.yaml`:
```yaml
memory_search:
  enabled: true
  db_path: ".deer-flow/data/search.db"
  max_results: 3
  max_content_chars: 100000
```

---

## 4.6 Files Summary

### New Files

| File | Lines (est.) | Description |
|------|-------------|-------------|
| `deerflow/memory_search/__init__.py` | ~5 | Module exports |
| `deerflow/memory_search/storage.py` | ~120 | Database init, schema, triggers, CRUD |
| `deerflow/memory_search/search.py` | ~200 | FTS5 search, query sanitize, smart truncation, LLM summary |
| `deerflow/memory_search/tool.py` | ~80 | session_search LangChain tool |
| `deerflow/memory_search/indexing.py` | ~50 | Message write helper |
| `config/memory_search_config.py` | ~30 | Config model |

### Modified Files

| File | Change |
|------|--------|
| `config/app_config.py` | Add `memory_search: MemorySearchConfig` field |
| `runtime/journal.py` | Accept optional `search_indexer`, call in `_flush_async()` |
| `agents/lead_agent/agent.py` | Register `session_search` tool |
| `config.example.yaml` | Add `memory_search` section |

### New Test Files

| File | Tests |
|------|-------|
| `tests/test_memory_search_storage.py` | Table creation, triggers, CRUD, dual FTS5 |
| `tests/test_memory_search_search.py` | Query sanitize, search logic, smart truncation |
| `tests/test_memory_search_tool.py` | Tool integration, both modes |

---

## 4.7 Error Handling

- **Search database write failure**: Logged as warning, never crashes RunJournal
- **FTS5 query error**: Catch and return user-friendly error message
- **LLM summarization failure**: Return raw truncated content instead of summary
- **Search database locked**: SQLite WAL mode handles concurrent readers; single
  writer retries with short backoff
- **Database file missing**: Auto-created on first access via `storage.init_db()`

---

## 4.8 Out of Scope

- PostgreSQL full-text search support
- Session search API endpoint (only agent tool for now)
- Session lineage/delegation tracking (DeerFlow doesn't have Hermes's parent-child session model)
- Real-time search index updates for already-running sessions (only new messages after deployment)
- Historical data migration (new index starts empty, builds incrementally)
