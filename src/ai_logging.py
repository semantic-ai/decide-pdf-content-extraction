"""Thin wrappers around Task.record_ai_call() for ML and LLM inference."""

from typing import Any, Optional

from decide_ai_service_base.util import get_agent_uri


def record_ml_call(
    task: Any,
    endpoint: str,
    duration: float,
) -> None:
    """Record a classic ML inference call (no token usage)."""
    task.record_ai_call(
        endpoint=endpoint,
        model_uri=get_agent_uri(),
        tokens_in=0,
        tokens_out=0,
        duration=duration,
    )


def record_llm_call(
    task: Any,
    endpoint: str,
    response: Any,
    duration: float,
) -> None:
    """Record an LLM API call, extracting token counts from response metadata."""
    tokens_in: int = 0
    tokens_out: int = 0

    usage = getattr(response, "usage_metadata", None)
    if usage:
        tokens_in = usage.get("input_tokens", 0) or 0
        tokens_out = usage.get("output_tokens", 0) or 0

    task.record_ai_call(
        endpoint=endpoint,
        model_uri=get_agent_uri(),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration=duration,
    )
