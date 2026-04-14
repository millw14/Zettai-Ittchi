"""FastAPI application — the endpoint Cursor talks to."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .config import detected_api_keys, get_config, resolve_preset
from .debate import run_debate
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ConsensusOutcome,
    ResponseMessage,
    UsageInfo,
)
from .stream import stream_debate_debug, stream_final_answer
from .utils import generate_request_id, setup_logging, timestamp

logger = logging.getLogger("zettai.app")

_startup_time: float = 0.0

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    global _startup_time
    cfg = get_config()
    setup_logging(cfg.server.log_level)

    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    keys = detected_api_keys()
    if not keys:
        logger.warning(
            "No API keys detected!  Add keys to .env — the server will "
            "start but all model calls will fail."
        )

    _startup_time = time.time()
    logger.info(
        "Zettai Ittchi v%s starting on %s:%s",
        __version__, cfg.server.host, cfg.server.port,
    )
    logger.info("Agents: %s", [a.name for a in cfg.agents])
    logger.info("Detected API keys: %s", keys)
    logger.info("Output mode: %s  |  Strict unanimity: %s",
                cfg.output.default_mode, cfg.debate.strict_unanimity)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Zettai Ittchi",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global error handler — always return OpenAI-compat JSON
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": 500,
            }
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    cfg = get_config()
    return {
        "status": "ok",
        "version": __version__,
        "agents": [
            {"name": a.name, "model": a.model, "role": a.role}
            for a in cfg.agents
        ],
        "detected_api_keys": detected_api_keys(),
        "strict_unanimity": cfg.debate.strict_unanimity,
        "output_mode": cfg.output.default_mode,
        "startup_time": _startup_time,
    }


@app.get("/v1/models")
async def list_models():
    cfg = get_config()
    models: list[dict[str, Any]] = []
    all_names = [cfg.virtual_models.default] + list(cfg.virtual_models.presets.keys())
    for name in all_names:
        models.append({
            "id": name,
            "object": "model",
            "created": int(_startup_time),
            "owned_by": "zettai-ittchi",
        })
    return {"object": "list", "data": models}


def _format_answer(outcome: ConsensusOutcome, mode: str) -> str:
    """Build the final text the user sees, based on output mode."""
    answer = outcome.final_answer

    if mode == "ghost":
        return answer

    if mode == "clean":
        footer = (
            f"\n\n[Consensus: {outcome.consensus_status} "
            f"after {outcome.rounds_used} round(s)]"
        )
        return answer + footer

    if mode == "audit":
        meta_lines = [
            f"\n\n---",
            f"**Consensus:** {outcome.consensus_status}",
            f"**Rounds:** {outcome.rounds_used}",
            f"**Latency:** {outcome.total_latency_ms:.0f}ms",
            f"**Estimated cost:** ${outcome.estimated_cost:.4f}",
        ]
        if outcome.unresolved_objections:
            meta_lines.append("**Unresolved objections:**")
            for obj in outcome.unresolved_objections:
                meta_lines.append(f"  - {obj}")
        return answer + "\n".join(meta_lines)

    # "debug" mode is handled via streaming, but if non-streaming we
    # fall back to the stream formatter's text output
    return answer


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    cfg = resolve_preset(request.model)

    # Optional header override for manual testing
    header_mode = raw_request.headers.get("x-zettai-mode")
    output_mode = header_mode if header_mode in ("ghost", "clean", "debug", "audit") else cfg.output.default_mode

    rid = generate_request_id()
    model_name = request.model
    ts = timestamp()

    logger.info(
        "Request %s  model=%s  stream=%s  mode=%s  messages=%d",
        rid, model_name, request.stream, output_mode, len(request.messages),
    )

    # Run the debate
    outcome = await run_debate(request.messages, cfg)

    logger.info(
        "Request %s  result=%s  rounds=%d  latency=%.0fms  cost=$%.4f",
        rid, outcome.consensus_status, outcome.rounds_used,
        outcome.total_latency_ms, outcome.estimated_cost,
    )

    # Streaming response
    if request.stream:
        if output_mode == "debug":
            generator = stream_debate_debug(outcome, model_name, rid)
        else:
            text = _format_answer(outcome, output_mode)
            generator = stream_final_answer(text, model_name, rid)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
        )

    # Non-streaming response
    if output_mode == "debug":
        from .stream import _format_transcript
        final_text = _format_transcript(outcome)
    else:
        final_text = _format_answer(outcome, output_mode)

    return ChatCompletionResponse(
        id=rid,
        created=ts,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ResponseMessage(content=final_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(),
    )
