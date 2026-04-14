"""Prompt templates for every phase of the debate cycle.

All functions return plain strings.  The debate engine injects them as
system or user messages when calling each agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import CandidateAnswer, CritiqueResult, DraftResult, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_messages(messages: list[Message]) -> str:
    """Render the incoming chat messages into a readable block."""
    parts: list[str] = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        parts.append(f"[{m.role}]: {content}")
    return "\n".join(parts)


def _format_drafts(drafts: list[DraftResult]) -> str:
    sections: list[str] = []
    for d in drafts:
        sections.append(f"--- Draft by {d.agent_name} ({d.model}) ---\n{d.content}")
    return "\n\n".join(sections)


def _format_critiques(critiques: list[CritiqueResult]) -> str:
    sections: list[str] = []
    for c in critiques:
        sections.append(
            f"--- Critique by {c.agent_name} ({c.model}) ---\n{c.critique_text}"
        )
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# System prompt — shared identity for all debate agents
# ---------------------------------------------------------------------------

def system_prompt(role: str) -> str:
    return (
        "You are a debate participant in a structured consensus process. "
        f"Your assigned role is: {role}.\n\n"
        "Rules:\n"
        "- You are NOT an assistant serving the user directly.\n"
        "- You must critique weak reasoning, vague claims, and missing details.\n"
        "- Do NOT pretend consensus exists when it does not.\n"
        "- Be precise, direct, and constructive.\n"
        "- Support your positions with clear reasoning."
    )


# ---------------------------------------------------------------------------
# Phase 1 — Draft
# ---------------------------------------------------------------------------

def draft_user_prompt(messages: list[Message]) -> str:
    conversation = _format_messages(messages)
    return (
        "Below is a user request.  Answer it independently based on your "
        "expertise and assigned role.  Provide a thorough, well-structured "
        "response.\n\n"
        f"{conversation}"
    )


# ---------------------------------------------------------------------------
# Phase 2 — Critique
# ---------------------------------------------------------------------------

def critique_prompt(
    messages: list[Message],
    all_drafts: list[DraftResult],
) -> str:
    conversation = _format_messages(messages)
    drafts_block = _format_drafts(all_drafts)
    return (
        "Below is the original user request followed by draft answers from "
        "all debate participants.\n\n"
        f"=== USER REQUEST ===\n{conversation}\n\n"
        f"=== DRAFTS ===\n{drafts_block}\n\n"
        "Your task:\n"
        "1. Identify factual issues or inaccuracies in any draft.\n"
        "2. Point out missing constraints or edge cases.\n"
        "3. Suggest better framing or structure where appropriate.\n"
        "4. Flag implementation risks or incorrect assumptions.\n\n"
        "Be specific.  Reference which draft you are critiquing."
    )


# ---------------------------------------------------------------------------
# Phase 3 — Synthesize
# ---------------------------------------------------------------------------

def synthesize_prompt(
    messages: list[Message],
    drafts: list[DraftResult],
    critiques: list[CritiqueResult],
) -> str:
    conversation = _format_messages(messages)
    drafts_block = _format_drafts(drafts)
    critiques_block = _format_critiques(critiques)
    return (
        "You are the synthesizer.  Your job is to produce a single, "
        "best-possible answer to the user request by combining the strongest "
        "elements from all drafts and incorporating valid critiques.\n\n"
        f"=== USER REQUEST ===\n{conversation}\n\n"
        f"=== DRAFTS ===\n{drafts_block}\n\n"
        f"=== CRITIQUES ===\n{critiques_block}\n\n"
        "Produce ONLY the final candidate answer.  Do NOT include a vote, "
        "approval, or meta-commentary about the process."
    )


# ---------------------------------------------------------------------------
# Phase 4 — Vote
# ---------------------------------------------------------------------------

VOTE_JSON_SCHEMA = '{"approve": <bool>, "blocking_issues": [<string>, ...], "confidence": <float 0-1>}'


def vote_prompt(candidate: CandidateAnswer) -> str:
    return (
        "Below is a candidate answer produced by the synthesizer.  You must "
        "decide whether to APPROVE or BLOCK it.\n\n"
        f"=== CANDIDATE ANSWER (round {candidate.round_number}) ===\n"
        f"{candidate.content}\n\n"
        "Rules:\n"
        "- Approve ONLY if the candidate is acceptable with no blocking flaw.\n"
        "- Do NOT approve out of politeness or to speed up the process.\n"
        "- If you block, provide a short, actionable reason.\n\n"
        "You MUST return ONLY valid JSON in this exact shape:\n"
        f"{VOTE_JSON_SCHEMA}\n\n"
        "Return nothing else — no markdown fences, no explanation, just the "
        "JSON object."
    )


# ---------------------------------------------------------------------------
# Revision (used in rounds 2+)
# ---------------------------------------------------------------------------

def revise_prompt(candidate: CandidateAnswer, objections: list[str]) -> str:
    objections_block = "\n".join(f"- {o}" for o in objections)
    return (
        "The previous candidate answer was blocked.  Revise it to address "
        "the following objections while preserving its strengths.\n\n"
        f"=== CANDIDATE ANSWER (round {candidate.round_number}) ===\n"
        f"{candidate.content}\n\n"
        f"=== BLOCKING OBJECTIONS ===\n{objections_block}\n\n"
        "Produce ONLY the revised candidate answer.  No vote, no "
        "meta-commentary."
    )
