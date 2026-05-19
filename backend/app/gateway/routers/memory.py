"""Memory API router — Hermes-style MEMORY.md/USER.md storage."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from deerflow.agents.memory.store import MemoryStore
from deerflow.config.memory_config import get_memory_config
from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id

router = APIRouter(prefix="/api", tags=["memory"])


def _get_store(agent_name: str | None = None) -> MemoryStore:
    """Create a MemoryStore for the current user."""
    config = get_memory_config()
    paths = get_paths()
    user_id = get_effective_user_id()
    memory_dir = paths.user_dir(user_id) / (Path("agents") / agent_name / "memory" if agent_name else "memory")
    store = MemoryStore(memory_dir=memory_dir, memory_char_limit=config.memory_char_limit, user_char_limit=config.user_char_limit)
    store.load_from_disk()
    return store


class MemoryEntry(BaseModel):
    target: str = Field(..., description="memory or user")
    entries: list[str] = Field(default_factory=list)
    usage: str = Field(default="")


class MemoryDataResponse(BaseModel):
    memory: MemoryEntry = Field(default_factory=lambda: MemoryEntry(target="memory"))
    user: MemoryEntry = Field(default_factory=lambda: MemoryEntry(target="user"))


class MemoryConfigResponse(BaseModel):
    enabled: bool
    injection_enabled: bool
    storage_path: str
    memory_char_limit: int
    user_char_limit: int


class MemoryStatusResponse(BaseModel):
    config: MemoryConfigResponse
    data: MemoryDataResponse


class EntryActionRequest(BaseModel):
    action: str = Field(..., description="add, replace, or remove")
    target: str = Field(default="memory", description="memory or user")
    content: str | None = Field(default=None)
    old_text: str | None = Field(default=None)


@router.get("/memory", response_model=MemoryDataResponse)
async def get_memory():
    """Get current memory entries."""
    store = _get_store()
    mem_usage = store.get_usage("memory")
    usr_usage = store.get_usage("user")
    return MemoryDataResponse(
        memory=MemoryEntry(target="memory", entries=store.get_live_entries("memory"), usage=f"{mem_usage['chars']}/{mem_usage['limit']} chars"),
        user=MemoryEntry(target="user", entries=store.get_live_entries("user"), usage=f"{usr_usage['chars']}/{usr_usage['limit']} chars"),
    )


@router.post("/memory/reload", response_model=MemoryDataResponse)
async def reload_memory():
    """Force reload from disk."""
    return await get_memory()


@router.delete("/memory", response_model=MemoryDataResponse)
async def clear_memory():
    """Clear all memory entries."""
    store = _get_store()
    store.clear("memory")
    store.clear("user")
    return await get_memory()


@router.post("/memory/entries", response_model=MemoryDataResponse)
async def manage_entry(request: EntryActionRequest):
    """Add, replace, or remove a memory entry."""
    store = _get_store()
    if request.action == "add":
        if not request.content:
            raise HTTPException(status_code=400, detail="content is required for add action.")
        result = store.add(request.target, request.content)
    elif request.action == "replace":
        if not request.old_text or not request.content:
            raise HTTPException(status_code=400, detail="old_text and content are required for replace action.")
        result = store.replace(request.target, request.old_text, request.content)
    elif request.action == "remove":
        if not request.old_text:
            raise HTTPException(status_code=400, detail="old_text is required for remove action.")
        result = store.remove(request.target, request.old_text)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action '{request.action}'. Use: add, replace, remove.")

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error."))

    return await get_memory()


@router.get("/memory/config", response_model=MemoryConfigResponse)
async def get_memory_config_endpoint():
    """Get memory configuration."""
    config = get_memory_config()
    return MemoryConfigResponse(
        enabled=config.enabled,
        injection_enabled=config.injection_enabled,
        storage_path=config.storage_path,
        memory_char_limit=config.memory_char_limit,
        user_char_limit=config.user_char_limit,
    )


@router.get("/memory/status", response_model=MemoryStatusResponse)
async def get_memory_status():
    """Get memory status (config + data)."""
    config = get_memory_config()
    config_resp = MemoryConfigResponse(
        enabled=config.enabled,
        injection_enabled=config.injection_enabled,
        storage_path=config.storage_path,
        memory_char_limit=config.memory_char_limit,
        user_char_limit=config.user_char_limit,
    )
    return MemoryStatusResponse(config=config_resp, data=await get_memory())
