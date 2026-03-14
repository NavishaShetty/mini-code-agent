"""LiteLLM model abstraction for multi-provider LLM support.

This module wraps LiteLLM to provide a consistent interface for querying
different LLM providers (OpenAI, Anthropic, local models, etc.).

Interview talking points:
- Reasoning vs Non-Reasoning Models:
  - Use extended thinking (claude-3.5-sonnet) for planning phases
  - Use fast models (claude-3.5-haiku) for execution phases
  - Cost considerations: reasoning tokens are expensive

- Cost tracking is essential for production:
  - Batch similar requests
  - Cache common prompts
  - Use smaller models for simple tasks

- vLLM/KServe context:
  - High-throughput LLM serving with PagedAttention
  - Continuous batching for better throughput
  - Autoscaling based on load
"""

import logging
import os
from typing import Any

import litellm
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class LiteLLMConfig(BaseModel):
    """Configuration for the LiteLLM model."""

    model_name: str = "claude-sonnet-4-20250514"
    model_kwargs: dict[str, Any] = {}
    temperature: float = 0.0
    max_tokens: int = 4096
    cost_tracking: bool = True


class LiteLLMModel:
    """LiteLLM wrapper with retry logic and cost tracking.

    Supports any model provider that LiteLLM supports:
    - OpenAI: gpt-4, gpt-3.5-turbo
    - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
    - Local: ollama/*, vllm/*
    - And many more...
    """

    def __init__(self, *, config_class: type = LiteLLMConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("TOY_AGENT_RETRY_ATTEMPTS", "5"))),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.AuthenticationError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.NotFoundError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs) -> Any:
        """Internal query method with retry logic."""
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **(self.config.model_kwargs | kwargs),
        )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Query the model and return the response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            dict with 'content' (response text) and 'extra' (full response data)
        """
        # Clean messages to only include role and content
        clean_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

        response = self._query(clean_messages, **kwargs)

        # Track costs
        self.n_calls += 1
        if self.config.cost_tracking:
            try:
                cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
                self.cost += cost if cost > 0 else 0.0
            except Exception as e:
                logger.debug(f"Could not calculate cost: {e}")

        # Track tokens
        usage = getattr(response, "usage", None)
        if usage:
            self.total_input_tokens += getattr(usage, "prompt_tokens", 0)
            self.total_output_tokens += getattr(usage, "completion_tokens", 0)

        return {
            "content": response.choices[0].message.content or "",
            "extra": {
                "model": self.config.model_name,
                "usage": usage.model_dump() if usage else {},
                "cost": self.cost,
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Return usage statistics."""
        return {
            "n_calls": self.n_calls,
            "total_cost": round(self.cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    def get_template_vars(self) -> dict[str, Any]:
        """Return variables for template rendering."""
        return self.config.model_dump() | self.get_stats()
