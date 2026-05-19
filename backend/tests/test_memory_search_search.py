"""Tests for session search logic — query sanitization, CJK detection, truncation."""

from deerflow.memory_search.search import (
    contains_cjk,
    count_cjk,
    format_conversation,
    sanitize_fts5_query,
    truncate_around_matches,
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
        assert '"' not in result or "test" in result

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
        assert len(result) <= 600

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
