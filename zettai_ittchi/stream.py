"""SSE streaming helpers in OpenAI-compatible chunk format."""

from __future__ import annotations

from typing import AsyncGenerator

import orjson

from .schemas import ConsensusOutcome
from .utils import generate_request_id, timestamp

CHUNK_SIZE = 40


def _make_chunk(
    request_id: str,
    model: str,
    created: int,
    *,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
) -> bytes:
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    delta = chunk["choices"][0]["delta"]
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return b"data: " + orjson.dumps(chunk) + b"\n\n"


async def stream_final_answer(
    content: str,
    model_name: str,
    request_id: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Yield OpenAI-compatible SSE chunks for *content*."""
    rid = request_id or generate_request_id()
    ts = timestamp()

    # Initial chunk with role
    yield _make_chunk(rid, model_name, ts, role="assistant")

    # Content chunks
    for i in range(0, len(content), CHUNK_SIZE):
        yield _make_chunk(rid, model_name, ts, content=content[i : i + CHUNK_SIZE])

    # Final chunk
    yield _make_chunk(rid, model_name, ts, finish_reason="stop")
    yield b"data: [DONE]\n\n"


def _format_transcript(outcome: ConsensusOutcome) -> str:
    """Render the debate transcript into readable text for debug streaming."""
    parts: list[str] = []

    for r in outcome.transcript:
        parts.append(f"\n{'='*60}")
        parts.append(f"ROUND {r.round_number}")
        parts.append(f"{'='*60}\n")

        if r.drafts:
            parts.append("--- DRAFTS ---\n")
            for d in r.drafts:
                parts.append(f"[{d.agent_name} / {d.model}]")
                parts.append(d.content)
                parts.append("")

        if r.critiques:
            parts.append("--- CRITIQUES ---\n")
            for c in r.critiques:
                parts.append(f"[{c.agent_name} / {c.model}]")
                parts.append(c.critique_text)
                parts.append("")

        if r.candidate:
            parts.append("--- CANDIDATE ANSWER ---\n")
            parts.append(r.candidate.content)
            parts.append("")

        if r.votes:
            parts.append("--- VOTES ---\n")
            for v in r.votes:
                status = "APPROVE" if v.approve else "BLOCK"
                issues = ", ".join(v.blocking_issues) if v.blocking_issues else "none"
                parts.append(
                    f"[{v.agent_name}] {status} "
                    f"(confidence: {v.confidence:.2f}, issues: {issues})"
                )
            parts.append("")

    parts.append(f"\n{'='*60}")
    parts.append(f"RESULT: {outcome.consensus_status.upper()}")
    parts.append(f"Rounds: {outcome.rounds_used}  |  "
                 f"Latency: {outcome.total_latency_ms:.0f}ms  |  "
                 f"Cost: ${outcome.estimated_cost:.4f}")
    if outcome.unresolved_objections:
        parts.append("Unresolved objections:")
        for obj in outcome.unresolved_objections:
            parts.append(f"  - {obj}")
    parts.append(f"{'='*60}\n")
    parts.append("--- FINAL ANSWER ---\n")
    parts.append(outcome.final_answer)

    return "\n".join(parts)


async def stream_debate_debug(
    outcome: ConsensusOutcome,
    model_name: str,
    request_id: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Stream the full debate transcript as standard ``delta.content`` chunks."""
    text = _format_transcript(outcome)
    async for chunk in stream_final_answer(text, model_name, request_id):
        yield chunk
