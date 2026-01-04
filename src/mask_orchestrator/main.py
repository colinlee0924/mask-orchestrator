"""Main entry point for mask-orchestrator A2A server.

This server provides:
- A2A protocol endpoint (JSON-RPC) at /
- SSE streaming endpoint at /api/stream
"""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.routing import Mount

from mask.a2a import MaskA2AServer

from mask_orchestrator.agent import create_agent
from mask_orchestrator.streaming import router as streaming_router, set_agent

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


def create_app() -> FastAPI:
    """Create the FastAPI application with A2A and streaming endpoints."""
    # Setup tracing based on TRACING_BACKEND env var
    backend = os.environ.get("TRACING_BACKEND", "dual")

    if backend == "dual":
        from mask.observability import setup_dual_tracing

        print("Using DUAL tracing (Phoenix + Langfuse)...")
        setup_dual_tracing(project_name="mask-orchestrator")
    elif backend == "phoenix":
        from mask.observability import setup_openinference_tracing

        print("Using Phoenix tracing...")
        setup_openinference_tracing(
            project_name="mask-orchestrator",
            endpoint=os.environ.get(
                "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
            ),
        )
    elif backend == "langfuse":
        from mask.observability import setup_langfuse_otel_tracing

        print("Using Langfuse tracing...")
        setup_langfuse_otel_tracing()
    else:
        print("No tracing configured (set TRACING_BACKEND=dual|phoenix|langfuse)")

    # Create orchestrator agent
    agent = create_agent()

    # Set agent for streaming module
    set_agent(agent)

    # Create A2A server
    port = int(os.environ.get("PORT", "10020"))
    a2a_server = MaskA2AServer(
        agent=agent,
        name="mask-orchestrator",
        description="Multi-Agent Orchestrator - coordinates tasks across MASK agents",
    )

    # Get the A2A Starlette app
    a2a_app = a2a_server.get_app(host="0.0.0.0", port=port)

    # Create FastAPI app and mount A2A routes
    app = FastAPI(
        title="MASK Orchestrator",
        description="Multi-Agent Orchestrator with A2A and SSE streaming",
        version="0.1.0",
    )

    # Include streaming routes
    app.include_router(streaming_router)

    # Mount A2A app at root (for backward compatibility)
    # A2A routes: /, /.well-known/agent.json, etc.
    app.mount("/", a2a_app)

    return app


def main():
    """Start the server."""
    port = int(os.environ.get("PORT", "10020"))

    # Auto-discover agents from REMOTE_AGENT_URLS if configured
    remote_urls = os.environ.get("REMOTE_AGENT_URLS", "")
    if remote_urls:
        print(f"Remote agents configured: {remote_urls}")

    print(f"Starting mask-orchestrator on port {port}...")
    print(f"  A2A endpoint: http://localhost:{port}/")
    print(f"  SSE streaming: http://localhost:{port}/api/stream")

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
