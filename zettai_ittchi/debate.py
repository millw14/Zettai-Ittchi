"""Core debate orchestrator: draft → critique → synthesize → vote → decide."""

from __future__ import annotations

import logging
import time

from .config import AgentConfig, AppConfig
from .consensus import evaluate_votes, parse_vote, pick_fallback_answer
from .prompts import (
    critique_prompt,
    draft_user_prompt,
    revise_prompt,
    synthesize_prompt,
    system_prompt,
    vote_prompt,
)
from .providers import call_all_agents, call_model
from .schemas import (
    CandidateAnswer,
    ConsensusOutcome,
    CritiqueResult,
    DebateRound,
    DraftResult,
    Message,
    VoteResult,
)

logger = logging.getLogger("zettai.debate")


def _messages_to_dicts(messages: list[Message]) -> list[dict]:
    return [m.model_dump(exclude_none=True) for m in messages]


def _remaining_seconds(start: float, budget: float) -> float:
    return max(0.0, budget - (time.perf_counter() - start))


async def run_debate(
    messages: list[Message],
    config: AppConfig,
) -> ConsensusOutcome:
    """Run a full multi-round debate and return the consensus outcome."""

    agents = config.agents
    dc = config.debate
    start = time.perf_counter()
    total_cost = 0.0
    transcript: list[DebateRound] = []

    # ------------------------------------------------------------------
    # Phase 1: Draft — every agent answers independently in parallel
    # ------------------------------------------------------------------
    logger.info("Phase 1: Drafting with %d agents", len(agents))
    draft_messages = [
        [
            {"role": "system", "content": system_prompt(agent.role)},
            {"role": "user", "content": draft_user_prompt(messages)},
        ]
        for agent in agents
    ]

    draft_results_raw = await call_all_agents(
        agents,
        draft_messages,
        timeout_seconds=_remaining_seconds(start, dc.max_total_seconds),
    )

    drafts: list[DraftResult] = []
    for agent, (content, latency, cost) in zip(agents, draft_results_raw):
        drafts.append(DraftResult(
            agent_name=agent.name,
            model=agent.model,
            content=content,
            latency_ms=latency,
            cost=cost,
        ))
        total_cost += cost

    # ------------------------------------------------------------------
    # Phase 2: Critique — every agent sees all drafts
    # ------------------------------------------------------------------
    logger.info("Phase 2: Critiquing")
    critique_msgs = [
        [
            {"role": "system", "content": system_prompt(agent.role)},
            {"role": "user", "content": critique_prompt(messages, drafts)},
        ]
        for agent in agents
    ]

    critique_results_raw = await call_all_agents(
        agents,
        critique_msgs,
        timeout_seconds=_remaining_seconds(start, dc.max_total_seconds),
    )

    critiques: list[CritiqueResult] = []
    for agent, (content, latency, cost) in zip(agents, critique_results_raw):
        critiques.append(CritiqueResult(
            agent_name=agent.name,
            model=agent.model,
            critique_text=content,
            latency_ms=latency,
        ))
        total_cost += cost

    # ------------------------------------------------------------------
    # Phase 3: Synthesize — first agent builds candidate
    # ------------------------------------------------------------------
    logger.info("Phase 3: Synthesizing candidate answer")
    synthesizer = agents[0]
    synth_messages = [
        {"role": "system", "content": system_prompt(synthesizer.role)},
        {"role": "user", "content": synthesize_prompt(messages, drafts, critiques)},
    ]
    synth_content, synth_latency, synth_cost = await call_model(
        synthesizer.model,
        synth_messages,
        timeout_seconds=_remaining_seconds(start, dc.max_total_seconds),
    )
    total_cost += synth_cost

    candidate = CandidateAnswer(
        content=synth_content,
        synthesizer_model=synthesizer.model,
        round_number=1,
        source_summary=f"Synthesized from {len(drafts)} drafts and {len(critiques)} critiques",
    )

    # ------------------------------------------------------------------
    # Rounds: vote → decide → (optionally revise and re-vote)
    # ------------------------------------------------------------------
    for round_num in range(1, dc.max_rounds + 1):
        if _remaining_seconds(start, dc.max_total_seconds) <= 0:
            logger.warning("Time budget exhausted before round %d vote", round_num)
            break

        # Phase 4: Vote
        logger.info("Round %d: Voting on candidate", round_num)
        vote_msgs = [
            [
                {"role": "system", "content": system_prompt(agent.role)},
                {"role": "user", "content": vote_prompt(candidate)},
            ]
            for agent in agents
        ]

        vote_results_raw = await call_all_agents(
            agents,
            vote_msgs,
            timeout_seconds=_remaining_seconds(start, dc.max_total_seconds),
        )

        votes: list[VoteResult] = []
        for agent, (content, latency, cost) in zip(agents, vote_results_raw):
            vote = parse_vote(
                content,
                agent.name,
                agent.model,
                malformed_is_block=dc.malformed_vote_is_block,
            )
            votes.append(vote)
            total_cost += cost

        # Build transcript round
        debate_round = DebateRound(
            round_number=round_num,
            drafts=drafts if round_num == 1 else [],
            critiques=critiques if round_num == 1 else [],
            candidate=candidate,
            votes=votes,
        )
        transcript.append(debate_round)

        # Phase 5: Decision
        success, status, objections = evaluate_votes(
            votes, strict_unanimity=dc.strict_unanimity
        )

        if success:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Consensus reached: %s (round %d)", status, round_num)
            return ConsensusOutcome(
                success=True,
                consensus_status=status,
                final_answer=candidate.content,
                rounds_used=round_num,
                unresolved_objections=[],
                transcript=transcript,
                total_latency_ms=elapsed,
                estimated_cost=total_cost,
            )

        # Not yet — can we do another round?
        if round_num < dc.max_rounds and _remaining_seconds(start, dc.max_total_seconds) > 0:
            logger.info(
                "Round %d: %s — revising candidate (%d objections)",
                round_num, status, len(objections),
            )
            revise_msgs = [
                {"role": "system", "content": system_prompt(synthesizer.role)},
                {"role": "user", "content": revise_prompt(candidate, objections)},
            ]
            revised_content, rev_latency, rev_cost = await call_model(
                synthesizer.model,
                revise_msgs,
                timeout_seconds=_remaining_seconds(start, dc.max_total_seconds),
            )
            total_cost += rev_cost

            candidate = CandidateAnswer(
                content=revised_content,
                synthesizer_model=synthesizer.model,
                round_number=round_num + 1,
                source_summary=f"Revised to address {len(objections)} objections",
            )

    # ------------------------------------------------------------------
    # Fallback — max rounds or time exhausted
    # ------------------------------------------------------------------
    elapsed = (time.perf_counter() - start) * 1000
    _, final_status, final_objections = evaluate_votes(
        transcript[-1].votes if transcript else [],
        strict_unanimity=dc.strict_unanimity,
    )

    if dc.fallback_on_deadlock:
        fallback_answer = pick_fallback_answer(transcript)
    else:
        fallback_answer = candidate.content

    logger.info(
        "Debate ended without full consensus: %s after %d rounds",
        final_status, len(transcript),
    )

    return ConsensusOutcome(
        success=final_status == "near-consensus" and not dc.strict_unanimity,
        consensus_status=final_status,
        final_answer=fallback_answer,
        rounds_used=len(transcript),
        unresolved_objections=final_objections,
        transcript=transcript,
        total_latency_ms=elapsed,
        estimated_cost=total_cost,
    )
