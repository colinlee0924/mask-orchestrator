"""OpenAI-compatible API adapter for MASK Orchestrator.

This module provides an OpenAI-compatible API endpoint that wraps the
MASK Orchestrator A2A protocol, allowing integration with Open WebUI
and other OpenAI-compatible clients.

Supports structured output formatting with collapsible sections for
better UI rendering in Open WebUI.
"""

import os
import re
import uuid
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple
from contextlib import contextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from opentelemetry import trace

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# --- Tracing Setup (must be before tracer creation) ---
def _init_tracing():
    """Initialize tracing at module load time."""
    from mask.observability import setup_dual_tracing
    backend = os.environ.get("TRACING_BACKEND", "dual")
    if backend in ("dual", "phoenix"):
        setup_dual_tracing(project_name="openai-adapter")
        print("Tracing initialized for openai-adapter")

_init_tracing()
tracer = trace.get_tracer("openai-adapter")


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
ENABLE_DETAILS_FORMAT = os.environ.get("ENABLE_DETAILS_FORMAT", "true").lower() == "true"


# --- Response Formatting ---

def parse_structured_response(content: str) -> Tuple[str, str, str]:
    """Parse structured response from orchestrator.

    Extracts thinking, actions, and summary sections from the response.
    Returns (thinking, actions, summary) tuple.
    """
    thinking = ""
    actions = ""
    summary = ""

    # Try to extract **思考過程:** section
    thinking_match = re.search(
        r'\*\*思考過程:\*\*\s*(.*?)(?=\*\*執行動作:\*\*|\*\*結果摘要:\*\*|$)',
        content,
        re.DOTALL
    )
    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Try to extract **執行動作:** section
    actions_match = re.search(
        r'\*\*執行動作:\*\*\s*(.*?)(?=\*\*結果摘要:\*\*|$)',
        content,
        re.DOTALL
    )
    if actions_match:
        actions = actions_match.group(1).strip()

    # Try to extract **結果摘要:** section
    summary_match = re.search(
        r'\*\*結果摘要:\*\*\s*(.*?)$',
        content,
        re.DOTALL
    )
    if summary_match:
        summary = summary_match.group(1).strip()

    # If no structured sections found, return original content as summary
    if not thinking and not actions and not summary:
        return "", "", content

    return thinking, actions, summary


def count_tool_calls(actions: str) -> int:
    """Count number of tool calls in the actions section."""
    # Count "工具:" occurrences
    return len(re.findall(r'- 工具:', actions))


def format_response_with_details(content: str) -> str:
    """Format response with collapsible sections for Open WebUI.

    Uses <think> tags which Open WebUI renders as collapsible
    "思考用時" sections. The final answer appears below.
    """
    if not ENABLE_DETAILS_FORMAT:
        return content

    thinking, actions, summary = parse_structured_response(content)

    # If no structured content, return original
    if not thinking and not actions:
        return content

    parts = []

    # Combine thinking and actions into a <think> block
    # Open WebUI renders this as collapsible "思考用時 X 秒" section
    think_content = []

    if thinking:
        step_count = count_tool_calls(actions) if actions else 0
        if step_count > 0:
            think_content.append(f"**💭 分析** ({step_count} 個步驟)\n\n{thinking}")
        else:
            think_content.append(f"**💭 分析**\n\n{thinking}")

    if actions:
        think_content.append(f"**🔧 執行**\n\n{actions}")

    if think_content:
        parts.append(f"<think>\n{chr(10).join(think_content)}\n</think>")

    # Final answer (always visible)
    if summary:
        parts.append(summary)
    elif not thinking and not actions:
        parts.append(content)

    return "\n\n".join(parts)


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


async def stream_from_orchestrator_sse(
    message: str,
    model: str,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream response from orchestrator SSE endpoint in OpenAI format.

    Connects to the /api/stream SSE endpoint and converts AgentEvents
    to OpenAI streaming format. Uses intelligent formatting:
    - Wraps thinking/reasoning in <think> tags for collapsible rendering
    - Shows tool calls with status indicators
    - Only final answer is visible outside <think> block
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Track state for smart formatting
    in_think_block = False
    tool_calls_count = 0
    has_content = False
    buffer = ""  # Buffer to detect section markers
    found_summary = False  # Track if we've seen the summary section

    # Markers that indicate the final answer section
    SUMMARY_MARKERS = ["**結果摘要:**", "**結果摘要：**", "結果摘要:", "Final Answer:", "Answer:"]
    THINKING_MARKERS = ["**思考過程:**", "**思考過程：**", "思考過程:", "Thinking:"]

    # Patterns to strip from output after summary marker (handles split markers)
    import re
    MARKER_CLEANUP_PATTERN = re.compile(r'^[\*]*[結果摘要:：\*\s]*')

    def make_chunk(content: str, finish_reason: Optional[str] = None) -> str:
        """Create an OpenAI-format SSE chunk."""
        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason
            }]
        }
        return f"data: {json.dumps(data)}\n\n"

    def find_marker_position(text: str, markers: list) -> int:
        """Find the last position of any marker in text. Returns -1 if not found."""
        last_pos = -1
        for marker in markers:
            pos = text.rfind(marker)
            if pos > last_pos:
                last_pos = pos
        return last_pos

    def check_markers(text: str) -> tuple[bool, bool]:
        """Check if text contains thinking or summary markers.
        Returns (is_thinking, is_summary) based on which appears LAST in text.
        """
        thinking_pos = find_marker_position(text, THINKING_MARKERS)
        summary_pos = find_marker_position(text, SUMMARY_MARKERS)

        # If summary marker appears after thinking marker, we're in summary mode
        if summary_pos > thinking_pos:
            return (False, True)
        elif thinking_pos >= 0:
            return (True, False)
        return (False, False)

    stream_url = f"{ORCHESTRATOR_URL}/api/stream"

    # Build request body, only include session_id if provided
    request_body = {"message": message}
    if session_id:
        request_body["session_id"] = session_id

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                stream_url,
                json=request_body,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # Parse SSE event
                    if line.startswith("event:"):
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue

                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event_data.get("type", "")
                        event_name = event_data.get("name", "")
                        data = event_data.get("data", {})

                        # Handle different event types
                        if event_type == "agent_start":
                            pass

                        elif event_type == "tool_call_start":
                            tool_calls_count += 1
                            tool_input = data.get("input", {})
                            input_preview = json.dumps(tool_input, ensure_ascii=False)[:200]

                            if not in_think_block:
                                yield make_chunk("<think>\n")
                                in_think_block = True

                            yield make_chunk(
                                f"🔧 **{event_name}** executing...\n"
                                f"```json\n{input_preview}\n```\n\n"
                            )

                        elif event_type == "tool_call_end":
                            tool_output = data.get("output", "")
                            output_preview = tool_output[:500]
                            if len(tool_output) > 500:
                                output_preview += "..."

                            yield make_chunk(
                                f"✅ **{event_name}** done\n"
                                f"```\n{output_preview}\n```\n\n"
                            )

                        elif event_type == "text_delta":
                            delta = data.get("delta", "")
                            if delta:
                                has_content = True
                                buffer += delta

                                # Check for section markers
                                is_thinking, is_summary = check_markers(buffer)

                                if is_thinking and not in_think_block:
                                    # Start of thinking section - open think block
                                    yield make_chunk("<think>\n")
                                    in_think_block = True
                                    yield make_chunk(delta)

                                elif is_summary and in_think_block:
                                    # Found summary - close think block first
                                    yield make_chunk("\n</think>\n\n")
                                    in_think_block = False
                                    found_summary = True
                                    # Strip marker pattern from this delta (handles split markers)
                                    # e.g., "果摘要:**\n內容" -> "內容"
                                    cleaned = MARKER_CLEANUP_PATTERN.sub('', delta).lstrip('\n')
                                    if cleaned:
                                        yield make_chunk(cleaned)

                                elif is_summary and not in_think_block:
                                    # Summary without prior thinking, or continuing after summary
                                    if not found_summary:
                                        # First time seeing summary marker
                                        found_summary = True
                                        # Strip any marker remnants
                                        cleaned = MARKER_CLEANUP_PATTERN.sub('', delta).lstrip('\n')
                                        if cleaned:
                                            yield make_chunk(cleaned)
                                    else:
                                        # Already past summary, yield normally
                                        yield make_chunk(delta)

                                elif in_think_block:
                                    # Inside thinking block
                                    yield make_chunk(delta)

                                else:
                                    # No markers yet - start a think block for initial content
                                    if not found_summary:
                                        yield make_chunk("<think>\n")
                                        in_think_block = True
                                    yield make_chunk(delta)

                        elif event_type == "agent_end":
                            # Close any open think block
                            if in_think_block:
                                yield make_chunk("\n</think>\n\n")
                                in_think_block = False

                        elif event_type == "error":
                            error_msg = data.get("message", "Unknown error")
                            if in_think_block:
                                yield make_chunk("\n</think>\n\n")
                                in_think_block = False
                            yield make_chunk(f"\n❌ Error: {error_msg}\n")

        # If no content was emitted from streaming, fall back to A2A
        if not has_content:
            raw_response = await send_to_orchestrator(message)
            formatted = format_response_with_details(raw_response)
            yield make_chunk(formatted)

        # Send final chunk
        yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except httpx.ConnectError:
        # Fallback: try A2A endpoint if SSE not available
        yield make_chunk("⚠️ SSE streaming unavailable, using fallback...\n")
        raw_response = await send_to_orchestrator(message)
        formatted = format_response_with_details(raw_response)
        yield make_chunk(formatted)
        yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield make_chunk(f"\n❌ Stream error: {str(e)}\n")
        yield make_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"


async def stream_response(content: str, model: str) -> AsyncGenerator[str, None]:
    """Stream response in SSE format.

    Handles HTML tags (like <details>) as atomic units to prevent
    breaking them across chunks.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Split content into smart chunks that preserve HTML tags
    chunks = _split_content_for_streaming(content)

    for i, chunk in enumerate(chunks):
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


def _split_content_for_streaming(content: str) -> List[str]:
    """Split content into chunks while preserving HTML/XML tags.

    Sends <think>...</think> blocks as single chunks to ensure
    proper parsing by Open WebUI's collapsible section renderer.
    """
    chunks = []

    # Pattern to match <think>...</think> blocks
    think_pattern = re.compile(r'(<think>.*?</think>)', re.DOTALL)

    parts = think_pattern.split(content)

    for part in parts:
        if not part.strip():
            continue

        if part.startswith('<think>'):
            # Send entire think block as one chunk for proper rendering
            chunks.append(part)
        else:
            # Split regular text into smaller chunks for streaming effect
            words = part.split()
            chunk_size = 5  # words per chunk

            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunks and not chunks[-1].endswith('>'):
                    chunk = " " + chunk
                chunks.append(chunk)

    return chunks if chunks else [content]


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
    with tracer.start_as_current_span("chat_completions") as span:
        try:
            # Get the last user message
            user_messages = [m for m in request.messages if m.role == "user"]
            if not user_messages:
                raise HTTPException(status_code=400, detail="No user message found")

            last_message = user_messages[-1].content
            span.set_attribute("input.user_message", last_message[:500])

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

            # For streaming requests, use SSE directly (skip A2A call)
            if request.stream:
                span.set_attribute("stream.mode", "sse")
                return StreamingResponse(
                    stream_from_orchestrator_sse(
                        message=full_message,
                        model=request.model,
                    ),
                    media_type="text/event-stream",
                )

            # Non-streaming: use A2A protocol
            with tracer.start_as_current_span("call_orchestrator") as orch_span:
                orch_span.set_attribute("orchestrator.url", ORCHESTRATOR_URL)
                raw_response = await send_to_orchestrator(full_message)
                orch_span.set_attribute("orchestrator.response_length", len(raw_response))
                orch_span.set_attribute("orchestrator.raw_response", raw_response[:1000])

            # Format response with collapsible sections for Open WebUI
            with tracer.start_as_current_span("format_response") as fmt_span:
                response_content = format_response_with_details(raw_response)
                fmt_span.set_attribute("format.input_length", len(raw_response))
                fmt_span.set_attribute("format.output_length", len(response_content))
                fmt_span.set_attribute("format.has_think_tag", "<think>" in response_content)
                fmt_span.set_attribute("format.transformed_response", response_content[:1000])

            span.set_attribute("output.response_length", len(response_content))

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
                    completion_tokens=len(raw_response.split()),
                    total_tokens=len(full_message.split()) + len(raw_response.split()),
                ),
            )

        except httpx.HTTPStatusError as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise HTTPException(
                status_code=502,
                detail=f"Orchestrator error: {str(e)}"
            )
        except httpx.ConnectError:
            span.set_attribute("error", True)
            span.set_attribute("error.message", "Connection failed")
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to orchestrator at {ORCHESTRATOR_URL}"
            )
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
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
