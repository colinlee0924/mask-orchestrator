#!/usr/bin/env python3
"""Detailed debug script for astream_events.

This script adds verbose logging to trace exactly what events are being
received and why text_delta events might not be emitted.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


async def debug_orchestrator_events():
    """Debug OrchestratorAgent's internal event processing."""
    print("=" * 60)
    print("Detailed Debug: OrchestratorAgent Event Processing")
    print("=" * 60)

    from mask_orchestrator.agent import create_agent
    from mask.core import AgentEvent

    agent = create_agent()

    # Manually replicate what astream_events does to debug
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from typing import List, Dict, Any

    message = "Say 'Hello' and nothing else."
    messages: List[BaseMessage] = [HumanMessage(content=message)]

    # Build state
    state = agent._build_state(messages, [], None)
    tools = agent._get_tools(state)
    callbacks = agent._get_callbacks(session_id=None)
    config: Dict[str, Any] = {
        "callbacks": callbacks,
        "run_name": agent.name,
    }

    # Get graph
    graph = agent._get_graph(tools)

    print(f"\nAgent: {agent.name}")
    print(f"Model: {type(agent.model)}")
    print(f"Tools: {[t.name for t in tools]}")
    print(f"Graph type: {type(graph)}")

    print("\n--- Raw graph.astream_events() output ---\n")

    event_count = 0
    chat_model_events = []

    async for event in graph.astream_events(
        {"messages": messages},
        config=config,
        version="v2",
    ):
        event_type = event.get("event", "unknown")
        event_name = event.get("name", "")
        event_data = event.get("data", {})

        # Log all events briefly
        print(f"[{event_count:3d}] {event_type}: {event_name}")

        # Detailed logging for chat model stream events
        if event_type == "on_chat_model_stream":
            chunk = event_data.get("chunk")
            print(f"      chunk type: {type(chunk)}")
            if chunk:
                print(f"      chunk.content type: {type(chunk.content)}")
                print(f"      chunk.content: {repr(chunk.content)[:100]}")

                content = chunk.content
                # Test the conditions
                print(f"      isinstance(content, str): {isinstance(content, str)}")
                print(f"      isinstance(content, list): {isinstance(content, list)}")
                print(f"      bool(content): {bool(content)}")

                chat_model_events.append({
                    "chunk_type": type(chunk).__name__,
                    "content_type": type(content).__name__,
                    "content": content,
                })

        event_count += 1

    print(f"\n--- Summary ---")
    print(f"Total events: {event_count}")
    print(f"on_chat_model_stream events: {len(chat_model_events)}")

    if chat_model_events:
        print("\n--- Chat Model Stream Events Analysis ---")
        for i, evt in enumerate(chat_model_events):
            print(f"\n  Event {i}:")
            print(f"    chunk type: {evt['chunk_type']}")
            print(f"    content type: {evt['content_type']}")
            print(f"    content: {repr(evt['content'])[:200]}")


async def debug_mask_astream_events():
    """Debug the actual MASK astream_events method."""
    print("\n" + "=" * 60)
    print("Debug: OrchestratorAgent.astream_events() output")
    print("=" * 60)

    from mask_orchestrator.agent import create_agent

    agent = create_agent()

    print("\n--- Calling agent.astream_events() ---\n")

    event_count = 0
    async for event in agent.astream_events("Say 'Hello' and nothing else."):
        print(f"[{event_count:3d}] {event.type}")
        print(f"      name: {event.name}")
        print(f"      data: {repr(event.data)[:100] if event.data else 'None'}")
        event_count += 1

    print(f"\n--- Total MASK events: {event_count} ---")


async def main():
    await debug_orchestrator_events()
    await debug_mask_astream_events()


if __name__ == "__main__":
    asyncio.run(main())
