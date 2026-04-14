"""Pure-logic consensus evaluation — no I/O."""

from __future__ import annotations

import json
import logging
from typing import Literal

from .schemas import CandidateAnswer, ConsensusOutcome, DebateRound, VoteResult

logger = logging.getLogger("zettai.consensus")


# ---------------------------------------------------------------------------
# Vote parsing
# ---------------------------------------------------------------------------

def parse_vote(
    raw_text: str,
    agent_name: str,
    model: str,
    *,
    malformed_is_block: bool = True,
) -> VoteResult:
    """Parse structured JSON vote output from an agent.

    If parsing fails and *malformed_is_block* is ``True`` the vote counts as
    a block — preventing silent failures from inflating unanimity.
    """
    cleaned = raw_text.strip()

    # Strip markdown code fences if the model wrapped its JSON
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        return VoteResult(
            agent_name=agent_name,
            model=model,
            approve=bool(data.get("approve", False)),
            blocking_issues=list(data.get("blocking_issues", [])),
            confidence=float(data.get("confidence", 0.0)),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Malformed vote from %s: %s — raw: %.200s",
            agent_name, exc, raw_text,
        )
        if malformed_is_block:
            return VoteResult(
                agent_name=agent_name,
                model=model,
                approve=False,
                blocking_issues=["Malformed vote output"],
                confidence=0.0,
            )
        return VoteResult(
            agent_name=agent_name,
            model=model,
            approve=True,
            blocking_issues=[],
            confidence=0.5,
        )


# ---------------------------------------------------------------------------
# Vote evaluation
# ---------------------------------------------------------------------------

def evaluate_votes(
    votes: list[VoteResult],
    *,
    strict_unanimity: bool = True,
) -> tuple[bool, Literal["unanimous", "near-consensus", "deadlock"], list[str]]:
    """Evaluate a list of votes and return ``(success, status, objections)``.

    Consensus rules:
    - All approve → ``unanimous``, success is always True.
    - Exactly one blocker → ``near-consensus``, success only if
      ``strict_unanimity`` is False.
    - Multiple blockers → ``deadlock``, success is always False.
    """
    blockers = [v for v in votes if not v.approve]
    all_objections: list[str] = []
    for v in blockers:
        all_objections.extend(v.blocking_issues)

    if len(blockers) == 0:
        return True, "unanimous", []

    if len(blockers) == 1:
        success = not strict_unanimity
        return success, "near-consensus", all_objections

    return False, "deadlock", all_objections


# ---------------------------------------------------------------------------
# Fallback candidate selection
# ---------------------------------------------------------------------------

def pick_fallback_answer(
    rounds: list[DebateRound],
) -> str:
    """Select the candidate with the highest average voter confidence.

    Iterates all completed rounds that have a candidate and votes, picks
    the one whose voters had the highest mean confidence.  Falls back to
    the last candidate if there are no votes at all.
    """
    best_content = ""
    best_avg = -1.0

    for r in rounds:
        if r.candidate is None:
            continue
        if not r.votes:
            best_content = r.candidate.content
            continue
        avg = sum(v.confidence for v in r.votes) / len(r.votes)
        if avg > best_avg:
            best_avg = avg
            best_content = r.candidate.content

    return best_content
