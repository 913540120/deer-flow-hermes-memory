"""Prompt formatting for memory injection into system prompt."""


def format_memory_block(memory_snapshot: str | None, user_snapshot: str | None) -> str:
    """Format frozen memory snapshots for system prompt injection.

    Args:
        memory_snapshot: Frozen snapshot from MemoryStore.format_for_system_prompt("memory").
        user_snapshot: Frozen snapshot from MemoryStore.format_for_system_prompt("user").

    Returns:
        Formatted memory block for system prompt, or empty string if both are empty.
    """
    parts = []
    if memory_snapshot:
        parts.append(memory_snapshot)
    if user_snapshot:
        parts.append(user_snapshot)

    if not parts:
        return ""

    return "\n\n".join(parts)
