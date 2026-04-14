"""Thin async wrapper around LiteLLM for multi-model fan-out."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

# Suppress litellm's own load_dotenv() which walks up to parent dirs and
# can crash on non-UTF-8 .env files outside this project.  Our config.py
# already loads the correct project-local .env.
import dotenv as _dotenv

_orig_load = _dotenv.load_dotenv
_dotenv.load_dotenv = lambda *a, **kw: None  # type: ignore[assignment]
import litellm  # noqa: E402
_dotenv.load_dotenv = _orig_load  # type: ignore[assignment]

from .config import AgentConfig

logger = logging.getLogger("zettai.providers")

# Suppress noisy LiteLLM info logs by default
litellm.suppress_debug_info = True


async def call_model(
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.7,
    timeout_seconds: float | None = None,
) -> tuple[str, float, float]:
    """Call a single model via LiteLLM.

    Returns ``(content, latency_ms, cost)``.  On failure the content is an
    error description string and cost is 0.
    """
    start = time.perf_counter()
    try:
        coro = litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        if timeout_seconds is not None:
            response = await asyncio.wait_for(coro, timeout=timeout_seconds)
        else:
            response = await coro

        elapsed_ms = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        return content, elapsed_ms, cost

    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning("Timeout calling %s after %.0f ms", model, elapsed_ms)
        return f"[ERROR: timeout after {elapsed_ms:.0f}ms]", elapsed_ms, 0.0

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("Error calling %s: %s", model, exc)
        return f"[ERROR: {exc}]", elapsed_ms, 0.0


async def call_all_agents(
    agents: list[AgentConfig],
    messages_per_agent: list[list[dict[str, Any]]],
    *,
    temperature: float = 0.7,
    timeout_seconds: float | None = None,
) -> list[tuple[str, float, float]]:
    """Fan out to all agents in parallel.

    *messages_per_agent* must have the same length as *agents*.
    Each element is the full message list for that agent's call.

    Returns a list of ``(content, latency_ms, cost)`` tuples in agent order.
    Failed calls return an error string as content — they never crash the
    batch.
    """
    assert len(agents) == len(messages_per_agent)

    tasks = [
        call_model(
            agent.model,
            msgs,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        for agent, msgs in zip(agents, messages_per_agent)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    cleaned: list[tuple[str, float, float]] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.error(
                "Agent %s raised: %s", agents[i].name, result
            )
            cleaned.append((f"[ERROR: {result}]", 0.0, 0.0))
        else:
            cleaned.append(result)
    return cleaned
