import os
import time
from collections import OrderedDict
from typing import Any

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/chat")

# ---------------------------------------------------------------------------
# In-memory conversation store with 7-day TTL and LRU eviction
# ---------------------------------------------------------------------------
_SESSION_TTL = 7 * 24 * 3600  # 7 days in seconds
_MAX_SESSIONS = 10_000  # LRU eviction limit to prevent memory exhaustion
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB max base64 image size

# OrderedDict for LRU eviction — most recently used sessions move to the end
# Each entry: {"messages": list[dict], "last_active": float}
_sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_CONTEXT_TOKENS = 200_000  # Claude Sonnet context window

# ---------------------------------------------------------------------------
# Cached Anthropic client (created once, reused across requests)
# ---------------------------------------------------------------------------
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _evict_expired() -> None:
    """Remove sessions that have been inactive for longer than the TTL."""
    now = time.time()
    expired = [
        sid for sid, data in _sessions.items()
        if now - data["last_active"] > _SESSION_TTL
    ]
    for sid in expired:
        del _sessions[sid]


def _get_or_create_session(session_id: str) -> dict[str, Any]:
    _evict_expired()
    if session_id in _sessions:
        # Move to end (most recently used)
        _sessions.move_to_end(session_id)
    else:
        # Evict oldest sessions if at capacity
        while len(_sessions) >= _MAX_SESSIONS:
            _sessions.popitem(last=False)
        _sessions[session_id] = {"messages": [], "last_active": time.time()}
    session = _sessions[session_id]
    session["last_active"] = time.time()
    return session


def _estimate_tokens(messages: list[dict], system: str | None = None) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = 0
    if system:
        total_chars += len(system)
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
                    elif block.get("type") == "image":
                        # Base64 images use ~1 token per 750 pixels; estimate 1000 tokens
                        total_chars += 4000
    return total_chars // 4


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=100_000)
    image: str | None = Field(default=None, description="Optional base64-encoded image")
    page_context: str | None = Field(
        default=None, description="Optional current page content for context"
    )
    api_key: str | None = Field(
        default=None, description="Optional user-provided Anthropic API key"
    )
    model: str | None = Field(
        default=None, description="Optional model override"
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str


class CompactRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=256)


class CompactResponse(BaseModel):
    session_id: str
    original_messages: int
    compacted_messages: int


class ContextUsageResponse(BaseModel):
    session_id: str
    estimated_tokens: int
    max_tokens: int
    usage_percent: float
    message_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to Claude and get a response, with per-session history."""
    # Use user-provided API key if supplied, otherwise fall back to server key
    if request.api_key:
        client = anthropic.Anthropic(api_key=request.api_key)
    else:
        try:
            client = _get_client()
        except HTTPException:
            raise HTTPException(
                status_code=422,
                detail="No API key configured. Please open Settings (gear icon) in the QA toolbox and enter your Anthropic API key.",
            )

    model = request.model or DEFAULT_MODEL
    session = _get_or_create_session(request.session_id)

    # Build user message content
    content: list[dict[str, Any]] = []

    if request.image:
        if len(request.image) > _MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds maximum size of {_MAX_IMAGE_BYTES // (1024 * 1024)} MB",
            )
        # Detect media type from base64 header or default to png
        media_type = "image/png"
        image_data = request.image
        if request.image.startswith("data:"):
            # Strip data URI prefix: "data:image/jpeg;base64,..."
            header, image_data = request.image.split(",", 1)
            if "image/jpeg" in header or "image/jpg" in header:
                media_type = "image/jpeg"
            elif "image/gif" in header:
                media_type = "image/gif"
            elif "image/webp" in header:
                media_type = "image/webp"

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        })

    content.append({"type": "text", "text": request.message})

    # Add user message to history
    session["messages"].append({"role": "user", "content": content})

    # Build system prompt
    system: str | None = None
    if request.page_context:
        system = (
            "The user is currently viewing the following page content. "
            "Use it as context when answering:\n\n"
            f"{request.page_context}"
        )

    # Call Claude API
    try:
        api_response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system or anthropic.NOT_GIVEN,
            messages=session["messages"],
        )
    except anthropic.AuthenticationError:
        session["messages"].pop()
        raise HTTPException(status_code=401, detail="Invalid API key. Please check your Anthropic API key in settings.")
    except anthropic.APIError as e:
        # Remove the failed user message from history
        session["messages"].pop()
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")

    assistant_text = api_response.content[0].text

    # Add assistant response to history
    session["messages"].append({"role": "assistant", "content": assistant_text})

    return ChatResponse(response=assistant_text, session_id=request.session_id)


@router.post("/compact", response_model=CompactResponse)
async def compact(request: CompactRequest) -> CompactResponse:
    """Summarise and compact the conversation history for a session."""
    client = _get_client()

    if request.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _get_or_create_session(request.session_id)
    original_count = len(session["messages"])

    if original_count == 0:
        return CompactResponse(
            session_id=request.session_id,
            original_messages=0,
            compacted_messages=0,
        )

    # Ask Claude to summarise the conversation
    summary_messages = session["messages"] + [
        {
            "role": "user",
            "content": (
                "Please provide a detailed summary of our entire conversation above. "
                "Capture all key points, decisions, code discussed, and any important context. "
                "This summary will replace the conversation history to save space."
            ),
        }
    ]

    try:
        api_response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=4096,
            messages=summary_messages,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")

    summary_text = api_response.content[0].text

    # Replace history with compact summary
    session["messages"] = [
        {
            "role": "user",
            "content": "Here is a summary of our previous conversation:\n\n" + summary_text,
        },
        {
            "role": "assistant",
            "content": (
                "I have the context from our previous conversation. "
                "How can I help you next?"
            ),
        },
    ]

    return CompactResponse(
        session_id=request.session_id,
        original_messages=original_count,
        compacted_messages=2,
    )


@router.get("/context-usage/{session_id}", response_model=ContextUsageResponse)
async def context_usage(session_id: str) -> ContextUsageResponse:
    """Return approximate token count and context usage for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _get_or_create_session(session_id)
    estimated = _estimate_tokens(session["messages"])
    usage_pct = round((estimated / MAX_CONTEXT_TOKENS) * 100, 2)

    return ContextUsageResponse(
        session_id=session_id,
        estimated_tokens=estimated,
        max_tokens=MAX_CONTEXT_TOKENS,
        usage_percent=usage_pct,
        message_count=len(session["messages"]),
    )
