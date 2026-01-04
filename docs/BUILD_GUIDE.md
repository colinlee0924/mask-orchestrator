# Mask Orchestrator 建置指南

本文檔記錄 `mask-orchestrator` 的完整建置過程，供未來開發者參考。

## 概述

`mask-orchestrator` 是一個基於 MASK Kernel 的多 Agent 協調者，整合了：
- **MASK Kernel**: 提供 Agent 基礎架構和 A2A 協議支援
- **A2A Protocol**: Agent-to-Agent 通訊協定
- **Dual Tracing**: 同時支援 Phoenix 和 Langfuse 可觀測性

## 專案結構

```
mask-orchestrator/
├── config/
│   └── prompts/
│       └── system.md             # Orchestrator 系統提示
├── src/mask_orchestrator/
│   ├── __init__.py               # 模組入口
│   ├── agent.py                  # Orchestrator Agent 實作
│   ├── main.py                   # A2A Server 啟動點
│   └── tools/
│       ├── __init__.py           # Tools 模組
│       └── orchestration.py      # 協調工具實作
├── tests/
│   └── test_orchestration.py     # 整合測試
├── .env.example                  # 環境變數範本
├── .gitignore
├── pyproject.toml                # Python 專案配置
└── README.md
```

## 建置步驟

### Step 1: 建立專案目錄

```bash
mkdir -p mask-orchestrator/config/prompts
mkdir -p mask-orchestrator/src/mask_orchestrator/tools
mkdir -p mask-orchestrator/tests
cd mask-orchestrator
```

### Step 2: 建立 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mask-orchestrator"
version = "0.1.0"
description = "Multi-Agent Orchestrator with MASK and A2A protocol"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Colin Lee", email = "colinlee0924@gmail.com" }
]
keywords = ["mask", "orchestrator", "agent", "a2a", "multi-agent"]

dependencies = [
    # 引用 mask-kernel 並啟用 phoenix, anthropic 兩個 extras
    "mask-kernel[phoenix,anthropic] @ git+https://github.com/colinlee0924/mask-kernel.git",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
]

[project.scripts]
mask-orchestrator = "mask_orchestrator.main:main"

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["src/mask_orchestrator"]
```

**重點說明**:
- `mask-kernel[phoenix,anthropic]` 啟用必要功能
- Orchestrator **不需要 MCP**，因為它透過 A2A 協定與其他 Agent 溝通
- `httpx` 用於發送 A2A 請求

### Step 3: 撰寫協調工具

建立 `src/mask_orchestrator/tools/__init__.py`:

```python
"""Orchestration tools package."""

from mask_orchestrator.tools.orchestration import get_orchestration_tools

__all__ = ["get_orchestration_tools"]
```

建立 `src/mask_orchestrator/tools/orchestration.py`:

```python
"""Orchestration tools for multi-agent coordination."""

import uuid
from typing import Dict, List, Optional

import httpx
from langchain_core.tools import tool

# Global registry to store discovered agents
_agent_registry: Dict[str, dict] = {}


def _fetch_agent_card(url: str) -> Optional[dict]:
    """Fetch agent card from a remote agent.

    Args:
        url: Base URL of the agent (e.g., http://localhost:10010)

    Returns:
        Agent card dict or None if fetch failed
    """
    try:
        agent_url = url.rstrip("/")
        with httpx.Client(timeout=10.0) as client:
            # A2A 標準端點
            response = client.get(f"{agent_url}/.well-known/agent.json")
            response.raise_for_status()
            return response.json()
    except Exception:
        # Fallback to root endpoint
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(agent_url)
                if response.status_code == 200:
                    data = response.json()
                    if "name" in data and "description" in data:
                        return data
        except Exception:
            pass
        return None


def _send_message_to_agent(url: str, message: str) -> str:
    """Send a message to a remote agent via A2A protocol."""
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

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=request_payload)
            response.raise_for_status()
            result = response.json()

            # Extract response text from A2A response
            if "result" in result and "parts" in result["result"]:
                texts = []
                for part in result["result"]["parts"]:
                    if part.get("kind") == "text" and "text" in part:
                        texts.append(part["text"])
                return "\n".join(texts) if texts else str(result)
            return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def discover_agents(urls: str) -> str:
    """Discover remote agents from comma-separated URLs.

    Args:
        urls: Comma-separated list of agent URLs
              (e.g., "http://localhost:10010,http://localhost:10030")

    Returns:
        Summary of discovered agents
    """
    global _agent_registry

    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    discovered = []
    failed = []

    for url in url_list:
        card = _fetch_agent_card(url)
        if card:
            agent_name = card.get("name", "unknown")
            _agent_registry[agent_name] = {
                "url": url,
                "card": card,
            }
            discovered.append(
                f"- {agent_name} ({url}): {card.get('description', 'No description')}"
            )
        else:
            failed.append(url)

    result = []
    if discovered:
        result.append(f"Successfully discovered {len(discovered)} agent(s):")
        result.extend(discovered)
    if failed:
        result.append(f"\nFailed to reach {len(failed)} URL(s): {', '.join(failed)}")

    return "\n".join(result) if result else "No agents discovered"


@tool
def list_all_agents() -> str:
    """List all discovered agents and their capabilities.

    Returns:
        Formatted list of all known agents
    """
    global _agent_registry

    if not _agent_registry:
        return "No agents discovered yet. Use discover_agents first."

    result = [f"Known agents ({len(_agent_registry)}):"]
    for name, info in _agent_registry.items():
        card = info.get("card", {})
        url = info.get("url", "unknown")
        description = card.get("description", "No description")
        skills = card.get("skills", [])

        result.append(f"\n## {name}")
        result.append(f"  URL: {url}")
        result.append(f"  Description: {description}")
        if skills:
            skill_names = [s.get("name", "unnamed") for s in skills]
            result.append(f"  Skills: {', '.join(skill_names)}")

    return "\n".join(result)


@tool
def send_task_to_agent(agent_name: str, task: str) -> str:
    """Send a task to a specific remote agent.

    Args:
        agent_name: Name of the agent to send the task to
        task: The task/message to send

    Returns:
        Agent's response
    """
    global _agent_registry

    if agent_name not in _agent_registry:
        available = list(_agent_registry.keys()) if _agent_registry else ["none"]
        return f"Agent '{agent_name}' not found. Available agents: {', '.join(available)}"

    info = _agent_registry[agent_name]
    url = info["url"]

    response = _send_message_to_agent(url, task)
    return f"Response from {agent_name}:\n{response}"


@tool
def broadcast_task(task: str) -> str:
    """Broadcast a task to all discovered agents.

    Args:
        task: The task/message to broadcast

    Returns:
        Combined responses from all agents
    """
    global _agent_registry

    if not _agent_registry:
        return "No agents discovered yet. Use discover_agents first."

    results = []
    for name, info in _agent_registry.items():
        url = info["url"]
        response = _send_message_to_agent(url, task)
        results.append(f"## {name}\n{response}")

    return "\n\n".join(results)


def get_orchestration_tools() -> List:
    """Get all orchestration tools."""
    return [
        discover_agents,
        list_all_agents,
        send_task_to_agent,
        broadcast_task,
    ]
```

**重點說明**:
- `_agent_registry` 使用 global dict 儲存發現的 agents
- `_fetch_agent_card()` 從 `/.well-known/agent.json` 取得 Agent 資訊
- `_send_message_to_agent()` 實作 A2A JSON-RPC 協定
- 四個工具：`discover_agents`、`list_all_agents`、`send_task_to_agent`、`broadcast_task`

### Step 4: 撰寫 System Prompt

建立 `config/prompts/system.md`:

```markdown
# Orchestrator Agent

You are a Multi-Agent Orchestrator. Your role is to coordinate tasks across multiple specialized agents.

## Available Tools
- discover_agents: Find remote agents by URL
- list_all_agents: See all available agents
- send_task_to_agent: Delegate task to specific agent
- broadcast_task: Send task to all agents

## Guidelines
- First discover agents using their URLs
- Analyze tasks and determine which agent is best suited
- Use send_task_to_agent for specific delegations
- Report results from each agent clearly
- Coordinate multi-step workflows by sequencing agent calls
```

### Step 5: 實作 Agent

建立 `src/mask_orchestrator/__init__.py`:

```python
"""mask-orchestrator: Multi-Agent Orchestrator with MASK."""
```

建立 `src/mask_orchestrator/agent.py`:

```python
"""Orchestrator Agent implementation using MASK kernel."""

from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool

from mask.agent import SimpleAgent, load_prompts
from mask.core import SkillRegistry
from mask.models import LLMFactory, ModelTier

from mask_orchestrator.tools import get_orchestration_tools


class OrchestratorAgent(SimpleAgent):
    """Multi-Agent Orchestrator.

    This agent coordinates tasks across multiple remote MASK agents using
    the A2A protocol. It provides tools for discovering agents, sending
    tasks, and orchestrating workflows.
    """

    def __init__(
        self,
        config_dir: str = "config",
        tier: ModelTier = ModelTier.THINKING,
    ):
        """Initialize the Orchestrator Agent.

        Args:
            config_dir: Path to configuration directory.
            tier: Model capability tier to use.
        """
        # Load prompts from config/prompts/
        prompts = load_prompts(config_dir)
        system_prompt = prompts.get(
            "system", "You are a Multi-Agent Orchestrator."
        )

        # Initialize LLM
        factory = LLMFactory()
        model = factory.get_model(tier=tier)

        # Initialize skill registry (supports future skill extensions)
        registry = SkillRegistry()

        # Discover skills from skills directory if it exists
        skills_dir = Path(__file__).parent / "skills"
        if skills_dir.exists():
            registry.discover_from_directory(skills_dir)

        # Get orchestration tools
        orchestration_tools: List[BaseTool] = get_orchestration_tools()

        super().__init__(
            model=model,
            skill_registry=registry,
            system_prompt=system_prompt,
            stateless=True,
            additional_tools=orchestration_tools,
        )


def create_agent(
    config_dir: str = "config",
    tier: ModelTier = ModelTier.THINKING,
) -> OrchestratorAgent:
    """Create and return the orchestrator agent instance."""
    return OrchestratorAgent(config_dir=config_dir, tier=tier)
```

**重點說明**:
- 繼承 `SimpleAgent`，自動獲得 `SkillMiddleware` 支援
- Orchestration tools 透過 `additional_tools` 傳入，永遠可見
- 支援未來擴充 skills（透過 `skills/` 目錄）

### Step 6: 實作 A2A Server

建立 `src/mask_orchestrator/main.py`:

```python
"""Main entry point for mask-orchestrator A2A server."""

import os
from pathlib import Path

from dotenv import load_dotenv

from mask.a2a import MaskA2AServer

from mask_orchestrator.agent import create_agent

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


def main():
    """Start the A2A server."""
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

    # Create A2A server
    server = MaskA2AServer(
        agent=agent,
        name="mask-orchestrator",
        description="Multi-Agent Orchestrator - coordinates tasks across MASK agents",
    )

    port = int(os.environ.get("PORT", "10020"))
    print(f"Starting mask-orchestrator on port {port}...")
    server.run(port=port)


if __name__ == "__main__":
    main()
```

### Step 7: 配置環境變數

建立 `.env` (從 `.env.example` 複製):

```bash
# LLM Provider
ANTHROPIC_API_KEY=your-anthropic-key

# Agent Port
PORT=10020

# Remote Agents (optional, for auto-discovery)
REMOTE_AGENT_URLS=http://localhost:10010,http://localhost:10030

# Tracing
TRACING_BACKEND=dual
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
LANGFUSE_SECRET_KEY=your-langfuse-secret
LANGFUSE_PUBLIC_KEY=your-langfuse-public
LANGFUSE_BASE_URL=http://localhost:3001
```

### Step 8: 安裝與執行

```bash
# 安裝依賴
uv sync

# 執行 Orchestrator
uv run mask-orchestrator
```

### Step 9: 測試協調功能

首先確保有其他 Agent 在運行（例如 jira-agent-mask 在 port 10010）:

```bash
# 測試 Agent 發現
curl -X POST http://localhost:10020/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "test-1",
        "role": "user",
        "parts": [{
          "kind": "text",
          "text": "Discover agents at http://localhost:10010 and list their capabilities"
        }]
      }
    }
  }'
```

```bash
# 測試任務委派
curl -X POST http://localhost:10020/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "2",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "test-2",
        "role": "user",
        "parts": [{
          "kind": "text",
          "text": "Send a task to jira-agent-mask: What tools do you have available?"
        }]
      }
    }
  }'
```

## 關鍵概念

### A2A Protocol 端點

每個 MASK Agent 都會暴露以下端點：

| 端點 | 方法 | 說明 |
|------|------|------|
| `/.well-known/agent.json` | GET | Agent Card（名稱、描述、技能） |
| `/` | POST | JSON-RPC 訊息處理 |

### Agent Card 結構

```json
{
  "name": "jira-agent-mask",
  "description": "Jira Agent with A2A support...",
  "skills": [...]
}
```

### Orchestration 工作流程

```
1. discover_agents("http://localhost:10010,http://localhost:10030")
   └─ Fetch agent cards, populate _agent_registry

2. list_all_agents()
   └─ Show all discovered agents and capabilities

3. send_task_to_agent("jira-agent-mask", "List open tickets")
   └─ Send A2A message, return response

4. broadcast_task("Status report")
   └─ Send same message to all agents
```

## 架構圖

```
                    ┌─────────────────────┐
                    │  mask-orchestrator  │ :10020
                    │   (中央協調者)       │
                    └──────────┬──────────┘
                               │ A2A Protocol
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │jira-agent-  │      │mask-deep-   │      │ Future...   │
   │   mask      │      │  agents     │      │             │
   │   :10010    │      │   :10030    │      │             │
   └─────────────┘      └─────────────┘      └─────────────┘
```

## 驗證 Traces

1. **Phoenix UI**: http://localhost:6006
   - 檢查 project: `mask-orchestrator`
   - Trace 結構: Agent → tool calls → remote agent responses

2. **Langfuse UI**: http://localhost:3001
   - 驗證 orchestration traces 是否完整記錄

## 常見問題

### Q: Agent 發現失敗？
A: 確認目標 Agent 已啟動，且 URL 正確（包含 port）

### Q: 工具呼叫逾時？
A: `_send_message_to_agent` 預設 60 秒逾時，可視需要調整

### Q: 如何新增更多協調工具？
A: 在 `orchestration.py` 中新增 `@tool` 裝飾的函數，並加入 `get_orchestration_tools()` 回傳列表

### Q: Agent Card 端點不正確？
A: A2A 標準使用 `/.well-known/agent.json`，程式已有 fallback 機制
