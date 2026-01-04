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
            response = client.get(f"{agent_url}/agent")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return None


def _send_message_to_agent(url: str, message: str) -> str:
    """Send a message to a remote agent via A2A protocol.

    Args:
        url: Base URL of the agent
        message: Message to send

    Returns:
        Agent's response text
    """
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
        urls: Comma-separated list of agent URLs (e.g., "http://localhost:10010,http://localhost:10030")

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
            discovered.append(f"- {agent_name} ({url}): {card.get('description', 'No description')}")
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
    """Get all orchestration tools.

    Returns:
        List of orchestration tool functions
    """
    return [
        discover_agents,
        list_all_agents,
        send_task_to_agent,
        broadcast_task,
    ]
