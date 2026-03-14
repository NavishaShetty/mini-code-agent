"""Prompt templates for the agent."""

from code_agent.prompts.system import (
    CODE_REVIEW_SYSTEM,
    DEFAULT_SYSTEM,
    DETAILED_SYSTEM,
    EXPLORER_SYSTEM,
    get_system_prompt,
)

__all__ = [
    "DEFAULT_SYSTEM",
    "DETAILED_SYSTEM",
    "CODE_REVIEW_SYSTEM",
    "EXPLORER_SYSTEM",
    "get_system_prompt",
]
