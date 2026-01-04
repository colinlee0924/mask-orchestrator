# MASK Orchestrator

A Multi-Agent Orchestrator that coordinates tasks across multiple MASK-based agents using the A2A protocol.

## Features

- **A2A Protocol Support**: Discover and communicate with remote agents
- **Dynamic Agent Discovery**: Find agents at runtime from configured URLs
- **Task Delegation**: Send tasks to specific agents or broadcast to all
- **Workflow Coordination**: Orchestrate multi-step workflows across agents
- **Dual Tracing**: Phoenix + Langfuse observability

## Architecture

```
┌─────────────────────────────────────────────────────┐
│             MASK Orchestrator                       │
│  ┌─────────────────────────────────────────┐       │
│  │            A2A Server                    │       │
│  │          (Port 10020)                    │       │
│  └─────────────────┬───────────────────────┘       │
│                    │                                │
│  ┌─────────────────▼───────────────────────┐       │
│  │         Orchestration Tools              │       │
│  │  - discover_agents                       │       │
│  │  - list_all_agents                       │       │
│  │  - send_task_to_agent                    │       │
│  │  - broadcast_task                        │       │
│  └─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
           │                    │
           ▼                    ▼
    ┌─────────────┐      ┌─────────────┐
    │ Agent A     │      │ Agent B     │
    │ :10010      │      │ :10030      │
    └─────────────┘      └─────────────┘
```

## Quick Start

### Prerequisites

1. **Remote Agents**: Other MASK agents running and accessible
2. **Python 3.10+**

### Installation

```bash
# Clone the repository
git clone https://github.com/colinlee0924/mask-orchestrator.git
cd mask-orchestrator

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Configuration

Edit `.env` file:

```bash
# LLM Provider
ANTHROPIC_API_KEY=your-anthropic-key

# Remote agents to discover on startup
REMOTE_AGENT_URLS=http://localhost:10010,http://localhost:10030

# Observability
TRACING_BACKEND=dual
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
LANGFUSE_SECRET_KEY=your-langfuse-secret
LANGFUSE_PUBLIC_KEY=your-langfuse-public
LANGFUSE_BASE_URL=http://localhost:3001
```

### Running

```bash
# Start the Orchestrator
uv run mask-orchestrator

# Or directly
uv run python -m mask_orchestrator.main
```

The orchestrator will start on port **10020**.

## API Examples

### A2A Message Format

```bash
curl -X POST http://localhost:10020/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "msg-001",
        "role": "user",
        "parts": [{"kind": "text", "text": "Discover agents at http://localhost:10010"}]
      }
    }
  }'
```

### Example Requests

- "Discover agents at http://localhost:10010,http://localhost:10030"
- "List all available agents"
- "Send task to jira-agent-mask: List my open tickets"
- "Broadcast to all agents: Report your status"

## License

MIT License - see [LICENSE](LICENSE) for details.
