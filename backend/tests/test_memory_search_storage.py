"""Tests for session search storage layer — SQLite schema, triggers, CRUD."""

from deerflow.memory_search.storage import SearchStorage


class TestSearchStorageInit:
    def test_creates_db_file_on_init(self, tmp_path):
        db_path = tmp_path / "test_search.db"
        assert not db_path.exists()
        SearchStorage(db_path)
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
        results = store.search_fts("secret", user_id="user1")
        assert len(results) == 1
        assert results[0]["session_id"] == "t1"
