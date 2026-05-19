"""Tests for StreamingMemoryScrubber state machine."""

from deerflow.agents.memory.scrubber import StreamingMemoryScrubber


class TestStreamingMemoryScrubberBasic:
    def test_clean_text_passes_through(self):
        s = StreamingMemoryScrubber()
        assert s.feed("Hello world") == "Hello world"

    def test_empty_string_returns_empty(self):
        s = StreamingMemoryScrubber()
        assert s.feed("") == ""

    def test_complete_memory_tag_removed(self):
        s = StreamingMemoryScrubber()
        text = "before<memory>secret data</memory>after"
        assert s.feed(text) == "beforeafter"

    def test_only_memory_tag_removed(self):
        s = StreamingMemoryScrubber()
        assert s.feed("<memory>everything</memory>") == ""

    def test_multiple_memory_tags_removed(self):
        s = StreamingMemoryScrubber()
        text = "a<memory>x</memory>b<memory>y</memory>c"
        assert s.feed(text) == "abc"

    def test_case_insensitive_tags(self):
        s = StreamingMemoryScrubber()
        text = "before<MEMORY>secret</MEMORY>after"
        assert s.feed(text) == "beforeafter"

    def test_multiline_content_removed(self):
        s = StreamingMemoryScrubber()
        text = "a<memory>line1\nline2\nline3</memory>b"
        assert s.feed(text) == "ab"


class TestStreamingMemoryScrubberSplit:
    def test_tag_split_across_two_feeds(self):
        s = StreamingMemoryScrubber()
        assert s.feed("hello<memo") == "hello"
        assert s.feed("ry>secret</memory>world") == "world"

    def test_close_tag_split_across_two_feeds(self):
        s = StreamingMemoryScrubber()
        assert s.feed("a<memory>secret") == "a"
        assert s.feed(" stuff</memo") == ""
        assert s.feed("ry>b") == "b"

    def test_tag_split_across_three_feeds(self):
        s = StreamingMemoryScrubber()
        assert s.feed("<mem") == ""
        assert s.feed("ory>hid") == ""
        assert s.feed("den</memory>visible") == "visible"

    def test_partial_tag_that_is_not_a_tag(self):
        s = StreamingMemoryScrubber()
        assert s.feed("hello<memo") == "hello"
        assert s.feed("randum>world") == "<memorandum>world"
        assert s.flush() == ""

    def test_text_held_back_for_potential_tag(self):
        s = StreamingMemoryScrubber()
        result = s.feed("hello<mem")
        assert result == "hello"
        result = s.feed("ory>secret</memory> world")
        assert result == " world"


class TestStreamingMemoryScrubberFlush:
    def test_flush_emits_held_partial_tag(self):
        s = StreamingMemoryScrubber()
        s.feed("hello<mem")
        assert s.flush() == "<mem"

    def test_flush_discards_inside_span(self):
        s = StreamingMemoryScrubber()
        s.feed("hello<memory>secret stuff")
        assert s.flush() == ""

    def test_flush_clears_state(self):
        s = StreamingMemoryScrubber()
        s.feed("hello<memory>secret")
        s.flush()
        assert s.feed("clean text") == "clean text"

    def test_flush_empty_buffer(self):
        s = StreamingMemoryScrubber()
        assert s.flush() == ""


class TestStreamingMemoryScrubberReset:
    def test_reset_clears_state(self):
        s = StreamingMemoryScrubber()
        s.feed("hello<memory>secret")
        s.reset()
        assert s.feed("clean") == "clean"

    def test_reset_then_flush(self):
        s = StreamingMemoryScrubber()
        s.feed("<memo")
        s.reset()
        assert s.flush() == ""


class TestStreamingMemoryScrubberEdgeCases:
    def test_adjacent_tags(self):
        s = StreamingMemoryScrubber()
        assert s.feed("<memory>a</memory><memory>b</memory>") == ""

    def test_tag_at_start(self):
        s = StreamingMemoryScrubber()
        assert s.feed("<memory>hidden</memory>visible") == "visible"

    def test_tag_at_end(self):
        s = StreamingMemoryScrubber()
        assert s.feed("visible<memory>hidden</memory>") == "visible"

    def test_empty_memory_tag(self):
        s = StreamingMemoryScrubber()
        assert s.feed("before<memory></memory>after") == "beforeafter"

    def test_no_held_text_when_no_partial_tag(self):
        s = StreamingMemoryScrubber()
        assert s.feed("complete text") == "complete text"
        assert s.flush() == ""

    def test_open_tag_only_never_closed(self):
        s = StreamingMemoryScrubber()
        assert s.feed("a<memory>b") == "a"
        assert s.flush() == ""
