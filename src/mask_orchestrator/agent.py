"""Orchestrator Agent implementation using MASK kernel."""

import os
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
    """Create and return the orchestrator agent instance.

    Args:
        config_dir: Path to configuration directory.
        tier: Model capability tier.

    Returns:
        Configured OrchestratorAgent instance.
    """
    return OrchestratorAgent(config_dir=config_dir, tier=tier)
