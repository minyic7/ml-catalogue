import os
import time
from collections import OrderedDict
from typing import Any

import anthropic
import openai
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
    model: str | None = Field(
        default=None, description="Optional model override"
    )
    # OpenAI-compatible endpoint settings (from user's browser localStorage)
    custom_base_url: str | None = Field(
        default=None, description="OpenAI-compatible API base URL"
    )
    custom_api_key: str | None = Field(
        default=None, description="API key for the custom endpoint"
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str


class CompactRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=256)
    api_key: str | None = Field(
        default=None, description="Optional user-provided Anthropic API key"
    )
    model: str | None = Field(
        default=None, description="Optional model override"
    )


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
# OpenAI-compatible call
# ---------------------------------------------------------------------------

def _call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    system: str | None,
) -> str:
    """Call an OpenAI-compatible endpoint and return the assistant text."""
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    # Convert Anthropic-style messages to OpenAI format
    oai_messages: list[dict[str, Any]] = []
    if system:
        oai_messages.append({"role": "system", "content": system})

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            oai_messages.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            # Convert Anthropic content blocks to OpenAI multi-part format
            parts: list[dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        })
            oai_messages.append({"role": msg["role"], "content": parts if parts else ""})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=oai_messages,
            max_tokens=4096,
        )
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key for custom endpoint.")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"Custom endpoint error: {e.message}")

    choice = response.choices[0]
    return choice.message.content or ""


# ---------------------------------------------------------------------------
# Anthropic call
# ---------------------------------------------------------------------------

def _call_anthropic(
    model: str,
    messages: list[dict],
    system: str | None,
) -> str:
    """Call Anthropic API using server-side key and return the assistant text."""
    client = _get_client()

    try:
        api_response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system or anthropic.NOT_GIVEN,
            messages=messages,
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Server Anthropic API key is invalid.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")

    if not api_response.content or not hasattr(api_response.content[0], "text"):
        raise HTTPException(status_code=502, detail="Claude API returned an empty or unexpected response")

    return api_response.content[0].text


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message and get a response. Uses custom OpenAI-compatible endpoint
    if provided, otherwise falls back to server-side Anthropic API key."""
    session = _get_or_create_session(request.session_id)

    # Build user message content
    content: list[dict[str, Any]] = []

    if request.image:
        if len(request.image) > _MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds maximum size of {_MAX_IMAGE_BYTES // (1024 * 1024)} MB",
            )
        media_type = "image/png"
        image_data = request.image
        if request.image.startswith("data:"):
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

    model = request.model or DEFAULT_MODEL

    try:
        if request.custom_base_url and request.custom_api_key:
            # Use user-provided OpenAI-compatible endpoint
            assistant_text = _call_openai_compatible(
                base_url=request.custom_base_url,
                api_key=request.custom_api_key,
                model=model,
                messages=session["messages"],
                system=system,
            )
        else:
            # Fallback to server-side Anthropic key
            assistant_text = _call_anthropic(
                model=model,
                messages=session["messages"],
                system=system,
            )
    except HTTPException:
        session["messages"].pop()
        raise

    # Add assistant response to history
    session["messages"].append({"role": "assistant", "content": assistant_text})

    return ChatResponse(response=assistant_text, session_id=request.session_id)


@router.post("/compact", response_model=CompactResponse)
async def compact(request: CompactRequest) -> CompactResponse:
    """Summarise and compact the conversation history for a session."""
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

    model = request.model or DEFAULT_MODEL

    try:
        api_response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=summary_messages,
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key. Please check your Anthropic API key in settings.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")

    if not api_response.content or not hasattr(api_response.content[0], "text"):
        raise HTTPException(status_code=502, detail="Claude API returned an empty or unexpected response")

    summary_text = api_response.content[0].text

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
