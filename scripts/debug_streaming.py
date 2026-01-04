#!/usr/bin/env python3
"""Debug script to test astream_events output from the orchestrator agent.

This script helps diagnose why on_chat_model_stream events may not be emitted.
Run with: cd mask-orchestrator && uv run python scripts/debug_streaming.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


async def test_raw_langgraph_events():
    """Test raw LangGraph astream_events to see what events are emitted."""
    print("\n" + "=" * 60)
    print("TEST 1: Raw LangGraph astream_events")
    print("=" * 60)

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent

    # Create a simple agent without tools
    model = ChatAnthropic(model="claude-sonnet-4-20250514")

    # Create simple agent
    agent = create_react_agent(model, tools=[])

    messages = [HumanMessage(content="Say 'Hello World' and nothing else.")]

    print("\nStreaming events from LangGraph agent...")
    event_count = 0
    event_types = {}

    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
    ):
        event_type = event.get("event", "unknown")
        event_types[event_type] = event_types.get(event_type, 0) + 1
        event_count += 1

        # Print interesting events
        if event_type in ["on_chat_model_stream", "on_tool_start", "on_tool_end"]:
            print(f"\n[{event_type}]")
            print(f"  name: {event.get('name', 'N/A')}")
            data = event.get("data", {})
            if event_type == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk:
                    content = getattr(chunk, "content", None)
                    print(f"  content: {repr(content)}")
            else:
                print(f"  data: {data}")

    print(f"\n\nTotal events: {event_count}")
    print("Event types summary:")
    for etype, count in sorted(event_types.items()):
        print(f"  {etype}: {count}")


async def test_orchestrator_astream_events():
    """Test OrchestratorAgent.astream_events()."""
    print("\n" + "=" * 60)
    print("TEST 2: OrchestratorAgent.astream_events()")
    print("=" * 60)

    from mask_orchestrator.agent import create_agent

    agent = create_agent()

    print("\nStreaming events from OrchestratorAgent...")
    event_count = 0
    event_types = {}

    async for event in agent.astream_events("Say 'Hello World' and nothing else."):
        event_types[event.type] = event_types.get(event.type, 0) + 1
        event_count += 1

        print(f"\n[{event.type}]")
        print(f"  name: {event.name}")
        if event.data:
            # Truncate long data
            data_str = str(event.data)
            if len(data_str) > 200:
                data_str = data_str[:200] + "..."
            print(f"  data: {data_str}")

    print(f"\n\nTotal events: {event_count}")
    print("Event types summary:")
    for etype, count in sorted(event_types.items()):
        print(f"  {etype}: {count}")


async def test_streaming_with_tools():
    """Test streaming with tool calls."""
    print("\n" + "=" * 60)
    print("TEST 3: Streaming with Tool Calls")
    print("=" * 60)

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny, 25°C"

    model = ChatAnthropic(model="claude-sonnet-4-20250514")
    agent = create_react_agent(model, tools=[get_weather])

    messages = [HumanMessage(content="What's the weather in Tokyo?")]

    print("\nStreaming events with tool calls...")
    event_count = 0
    event_types = {}
    text_content = []

    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
    ):
        event_type = event.get("event", "unknown")
        event_types[event_type] = event_types.get(event_type, 0) + 1
        event_count += 1

        if event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk:
                content = getattr(chunk, "content", None)
                if content:
                    text_content.append(content)
                    print(content, end="", flush=True)

        elif event_type == "on_tool_start":
            print(f"\n\n[TOOL START] {event.get('name')}")
            print(f"  Input: {event.get('data', {}).get('input')}")

        elif event_type == "on_tool_end":
            print(f"[TOOL END] {event.get('name')}")
            output = event.get("data", {}).get("output", "")
            print(f"  Output: {str(output)[:100]}...")

    print(f"\n\nTotal events: {event_count}")
    print("Event types summary:")
    for etype, count in sorted(event_types.items()):
        print(f"  {etype}: {count}")

    print(f"\nText content collected: {''.join(text_content)[:200]}...")


async def main():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("MASK Streaming Diagnostics")
    print("=" * 60)

    try:
        await test_raw_langgraph_events()
    except Exception as e:
        print(f"\nERROR in test 1: {e}")
        import traceback
        traceback.print_exc()

    try:
        await test_orchestrator_astream_events()
    except Exception as e:
        print(f"\nERROR in test 2: {e}")
        import traceback
        traceback.print_exc()

    try:
        await test_streaming_with_tools()
    except Exception as e:
        print(f"\nERROR in test 3: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
