"""FTS5 search logic — query sanitization, CJK detection, smart truncation, formatting."""

import re

MAX_SESSION_CHARS = 100_000

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x3000, 0x303F),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0xAC00, 0xD7AF),
)


def _is_cjk_codepoint(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def contains_cjk(text: str) -> bool:
    return any(_is_cjk_codepoint(ord(ch)) for ch in text)


def count_cjk(text: str) -> int:
    return sum(1 for ch in text if _is_cjk_codepoint(ord(ch)))


def sanitize_fts5_query(query: str) -> str:
    if not query or not query.strip():
        return ""

    quoted_parts: list[str] = []

    def _preserve_quoted(m: re.Match) -> str:
        quoted_parts.append(m.group(0))
        return f"\x00Q{len(quoted_parts) - 1}\x00"

    sanitized = re.sub(r'"[^"]*"', _preserve_quoted, query)
    sanitized = re.sub(r"[+{}()^]", " ", sanitized)
    sanitized = re.sub(r"\*+", "*", sanitized)
    sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)
    sanitized = sanitized.strip()
    sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized)
    sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized)
    sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)

    for i, quoted in enumerate(quoted_parts):
        sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

    return sanitized.strip()


def truncate_around_matches(full_text: str, query: str, max_chars: int = MAX_SESSION_CHARS) -> str:
    if len(full_text) <= max_chars:
        return full_text

    if not query or not query.strip():
        return full_text[:max_chars] + "\n\n...[truncated]..."

    text_lower = full_text.lower()
    query_lower = query.lower().strip()
    match_positions: list[int] = []

    phrase_pat = re.compile(re.escape(query_lower))
    match_positions = [m.start() for m in phrase_pat.finditer(text_lower)]

    if not match_positions:
        terms = query_lower.split()
        if len(terms) > 1:
            term_positions: dict[str, list[int]] = {}
            for t in terms:
                term_positions[t] = [m.start() for m in re.finditer(re.escape(t), text_lower)]
            rarest = min(terms, key=lambda t: len(term_positions.get(t, [])))
            for pos in term_positions.get(rarest, []):
                if all(any(abs(p - pos) < 200 for p in term_positions.get(t, [])) for t in terms if t != rarest):
                    match_positions.append(pos)

    if not match_positions:
        terms = query_lower.split()
        for t in terms:
            for m in re.finditer(re.escape(t), text_lower):
                match_positions.append(m.start())

    if not match_positions:
        truncated = full_text[:max_chars]
        suffix = "\n\n...[truncated]..." if max_chars < len(full_text) else ""
        return truncated + suffix

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


def format_conversation(messages: list[dict]) -> str:
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
    if ts is None:
        return "unknown"
    from datetime import UTC, datetime

    try:
        dt = datetime.fromtimestamp(ts, tz=UTC)
        return dt.strftime("%b %d, %Y at %I:%M %p UTC")
    except (OSError, ValueError):
        return "unknown"
