"""Tests for mask-orchestrator."""

import asyncio
import logging
from uuid import uuid4

import httpx


async def test_orchestration():
    """Test the orchestrator by discovering and coordinating with remote agents."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base_url = "http://localhost:10020"

    # Test discovering agents
    async with httpx.AsyncClient(timeout=60.0) as client:
        logger.info("Testing mask-orchestrator...")

        # Test 1: Discover jira-agent-mask
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid4()),
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Discover agents at http://localhost:10010",
                        }
                    ],
                }
            },
        }

        try:
            response = await client.post(base_url, json=request)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Discovery response: {result}")
        except Exception as e:
            logger.error(f"Discovery failed: {e}")

        # Test 2: List all agents
        request["id"] = str(uuid4())
        request["params"]["message"]["messageId"] = str(uuid4())
        request["params"]["message"]["parts"][0]["text"] = "List all available agents"

        try:
            response = await client.post(base_url, json=request)
            response.raise_for_status()
            result = response.json()
            logger.info(f"List agents response: {result}")
        except Exception as e:
            logger.error(f"List agents failed: {e}")

        logger.info("Test complete!")


if __name__ == "__main__":
    asyncio.run(test_orchestration())
