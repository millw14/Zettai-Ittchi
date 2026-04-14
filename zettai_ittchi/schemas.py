from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# OpenAI-compatible request types (broad field tolerance for Cursor)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list | None = None
    name: str | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """Accept every field Cursor might send; ignore what we don't use."""

    model_config = ConfigDict(extra="allow")

    messages: list[Message]
    model: str = "zettai-ittchi"
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    response_format: Any | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None


# ---------------------------------------------------------------------------
# OpenAI-compatible response types
# ---------------------------------------------------------------------------

class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ---------------------------------------------------------------------------
# OpenAI-compatible streaming chunk types
# ---------------------------------------------------------------------------

class ChatCompletionChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# ---------------------------------------------------------------------------
# Internal debate types
# ---------------------------------------------------------------------------

class DraftResult(BaseModel):
    agent_name: str
    model: str
    content: str
    latency_ms: float = 0.0
    cost: float = 0.0


class CritiqueResult(BaseModel):
    agent_name: str
    model: str
    critique_text: str
    latency_ms: float = 0.0


class CandidateAnswer(BaseModel):
    content: str
    synthesizer_model: str
    round_number: int
    source_summary: str = ""


class VoteResult(BaseModel):
    agent_name: str
    model: str
    approve: bool
    blocking_issues: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class DebateRound(BaseModel):
    round_number: int
    drafts: list[DraftResult] = Field(default_factory=list)
    critiques: list[CritiqueResult] = Field(default_factory=list)
    candidate: CandidateAnswer | None = None
    votes: list[VoteResult] = Field(default_factory=list)


class ConsensusOutcome(BaseModel):
    success: bool
    consensus_status: Literal["unanimous", "near-consensus", "deadlock"]
    final_answer: str
    rounds_used: int
    unresolved_objections: list[str] = Field(default_factory=list)
    transcript: list[DebateRound] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    estimated_cost: float = 0.0
