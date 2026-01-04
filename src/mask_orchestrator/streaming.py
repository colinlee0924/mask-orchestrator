"""SSE Streaming endpoint for mask-orchestrator.

This module provides Server-Sent Events (SSE) streaming for real-time
agent event delivery, enabling rich UI rendering in Open WebUI.
"""

import json
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from mask.agent.base_agent import BaseAgent

# Module-level agent reference (set by main.py)
_agent: "BaseAgent" = None


def set_agent(agent: "BaseAgent") -> None:
    """Set the agent instance for streaming."""
    global _agent
    _agent = agent


class StreamRequest(BaseModel):
    """Request body for streaming endpoint."""

    message: str
    session_id: str = None


# Create router for SSE endpoints
router = APIRouter(prefix="/api", tags=["streaming"])


@router.post("/stream")
async def stream_events(request: StreamRequest):
    """Stream agent events via Server-Sent Events (SSE).

    This endpoint provides real-time streaming of agent execution events,
    including tool calls, LLM responses, and agent lifecycle events.

    The events are formatted for consumption by the OpenAI adapter,
    which converts them to Open WebUI-compatible formats.

    Returns:
        EventSourceResponse: SSE stream of agent events.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def event_generator() -> AsyncGenerator[dict, None]:
        """Generate SSE events from agent execution."""
        try:
            async for event in _agent.astream_events(
                message=request.message,
                session_id=request.session_id,
            ):
                # Format event for SSE
                yield {
                    "event": event.type,
                    "data": json.dumps(event.to_dict()),
                }
        except Exception as e:
            # Emit error event
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "data": {"message": str(e)}}),
            }

    return EventSourceResponse(event_generator())


@router.get("/health")
async def health_check():
    """Health check endpoint for streaming service."""
    return {
        "status": "healthy",
        "agent_initialized": _agent is not None,
    }
