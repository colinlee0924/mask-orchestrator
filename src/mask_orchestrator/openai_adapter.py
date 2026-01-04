"""OpenAI-compatible API adapter for MASK Orchestrator.

This module provides an OpenAI-compatible API endpoint that wraps the
MASK Orchestrator A2A protocol, allowing integration with Open WebUI
and other OpenAI-compatible clients.
"""

import os
import uuid
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


app = FastAPI(
    title="MASK Orchestrator OpenAI Adapter",
    description="OpenAI-compatible API for MASK Orchestrator",
    version="0.1.0",
)


# --- Pydantic Models ---

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mask-orchestrator"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "mask"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# --- Configuration ---

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:10020")
TIMEOUT = int(os.environ.get("ADAPTER_TIMEOUT", "120"))


# --- Helper Functions ---

async def send_to_orchestrator(message: str) -> str:
    """Send message to MASK Orchestrator via A2A protocol."""
    request_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
            }
        },
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(ORCHESTRATOR_URL, json=request_payload)
        response.raise_for_status()
        result = response.json()

        # Extract response from A2A format
        if "result" in result and "parts" in result["result"]:
            texts = []
            for part in result["result"]["parts"]:
                if part.get("kind") == "text" and "text" in part:
                    texts.append(part["text"])
            return "\n".join(texts) if texts else json.dumps(result)

        if "error" in result:
            return f"Error: {result['error']}"

        return json.dumps(result)


async def stream_response(content: str, model: str) -> AsyncGenerator[str, None]:
    """Stream response in SSE format."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Split content into chunks for streaming effect
    words = content.split()
    chunk_size = 3  # words per chunk

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if i > 0:
            chunk = " " + chunk

        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"

    # Send final chunk
    final_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_data)}\n\n"
    yield "data: [DONE]\n\n"


# --- API Endpoints ---

@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models."""
    return ModelsResponse(
        data=[
            ModelInfo(id="mask-orchestrator"),
            ModelInfo(id="mask-orchestrator-multi-agent"),
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    """Get model info."""
    return ModelInfo(id=model_id)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    try:
        # Get the last user message
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        last_message = user_messages[-1].content

        # Build context from conversation history
        context_parts = []
        for msg in request.messages[:-1]:  # All messages except the last
            if msg.role == "system":
                context_parts.append(f"System: {msg.content}")
            elif msg.role == "assistant":
                context_parts.append(f"Assistant: {msg.content}")
            elif msg.role == "user":
                context_parts.append(f"User: {msg.content}")

        # If there's conversation history, include it
        if context_parts:
            full_message = (
                "Previous conversation:\n"
                + "\n".join(context_parts)
                + f"\n\nCurrent request: {last_message}"
            )
        else:
            full_message = last_message

        # Send to orchestrator
        response_content = await send_to_orchestrator(full_message)

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_response(response_content, request.model),
                media_type="text/event-stream",
            )

        # Return non-streaming response
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        return ChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                )
            ],
            usage=Usage(
                prompt_tokens=len(full_message.split()),
                completion_tokens=len(response_content.split()),
                total_tokens=len(full_message.split()) + len(response_content.split()),
            ),
        )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Orchestrator error: {str(e)}"
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to orchestrator at {ORCHESTRATOR_URL}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "orchestrator_url": ORCHESTRATOR_URL}


def main():
    """Run the OpenAI adapter server."""
    port = int(os.environ.get("ADAPTER_PORT", "8100"))
    print(f"Starting OpenAI adapter on port {port}...")
    print(f"Orchestrator URL: {ORCHESTRATOR_URL}")
    print(f"Connect Open WebUI to: http://localhost:{port}/v1")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
