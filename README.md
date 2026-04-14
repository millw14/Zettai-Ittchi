# Zettai Ittchi

**A local consensus proxy that turns multiple LLMs into one sharper answer.**

Zettai Ittchi sits between [Cursor](https://cursor.com) and your LLM providers. Every prompt triggers a structured debate across multiple models — they draft independently, critique each other, synthesize a candidate answer, and vote on it. You get back one clean response through a standard OpenAI-compatible endpoint. No database. No browser UI. Just `http://127.0.0.1:8000/v1`.

> **絶対一致** (zettai ittchi) — Japanese for "absolute agreement."

---

## How It Works

```
You (Cursor)                 Zettai Ittchi                    LLM Providers
    |                             |                                |
    |  POST /v1/chat/completions  |                                |
    |---------------------------->|                                |
    |                             |  1. Draft (all agents)         |
    |                             |------------------------------->|
    |                             |<-------------------------------|
    |                             |  2. Critique (all agents)      |
    |                             |------------------------------->|
    |                             |<-------------------------------|
    |                             |  3. Synthesize candidate       |
    |                             |------------------------------->|
    |                             |<-------------------------------|
    |                             |  4. Vote (structured JSON)     |
    |                             |------------------------------->|
    |                             |<-------------------------------|
    |                             |  5. Unanimous? Return answer   |
    |                             |     Blocked? Revise & re-vote  |
    |  One clean response         |                                |
    |<----------------------------|                                |
```

Every agent has a distinct role — Critical Analyst, Skeptical Reviewer, Creative Optimizer — so the debate has real tension, not performative agreement. Consensus is computed deterministically from structured votes, never by asking one model if everyone agrees.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/millw14/Zettai-Ittchi.git
cd Zettai-Ittchi
pip install -r requirements.txt
```

### 2. Add your API keys

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=xai-...
```

### 3. Start the server

```bash
python -m uvicorn zettai_ittchi.app:app --host 127.0.0.1 --port 8000
```

On Windows, double-click `start_zettai.bat` instead.

### 4. Connect Cursor

In Cursor settings, add a custom OpenAI-compatible endpoint:

- **Base URL:** `http://127.0.0.1:8000/v1`
- **Model name:** `zettai-ittchi`

Start in Ask mode. Chat normally. Every response is now debate-refined.

---

## Presets

Presets are exposed as model names — pick one from Cursor's model dropdown and the server adjusts behavior automatically. No config changes needed per-request.

| Model name | Rounds | Strict unanimity | Output mode | Best for |
|---|---|---|---|---|
| `zettai-ittchi` | 2 | Yes | ghost | Default — clean answers |
| `zettai-fast` | 1 | No | ghost | Speed — quick consensus |
| `zettai-balanced` | 2 | Yes | ghost | Reliability — solid consensus |
| `zettai-paranoid` | 4 | Yes | ghost | Rigor — exhaustive debate |
| `zettai-debug` | 2 | Yes | debug | Transparency — full transcript |

Add all five as models in Cursor and switch between them depending on the task.

---

## Output Modes

Control how much of the internal debate is visible in the response.

| Mode | What you see |
|---|---|
| **ghost** | Clean answer only. Debate is invisible. Feels like a single model. |
| **clean** | Clean answer with a one-line footer: `[Consensus: unanimous after 2 round(s)]` |
| **debug** | Full transcript — drafts, critiques, votes, then final answer |
| **audit** | Clean answer + metadata: models used, rounds, latency, cost, unresolved objections |

The default mode is set in `config.yaml` under `output.default_mode`. The `zettai-debug` preset overrides it to `debug`. For manual testing with `curl`, you can also pass `X-Zettai-Mode: debug` as a header.

---

## The Debate Protocol

### Phase 1 — Draft
Every agent receives the full conversation history and answers independently. No agent sees any other agent's work yet.

### Phase 2 — Critique
Every agent receives all drafts and must identify factual issues, missing constraints, better framing, and implementation risks.

### Phase 3 — Synthesize
The first agent combines the strongest elements from all drafts, incorporating valid critiques, into a single candidate answer.

### Phase 4 — Vote
Every agent evaluates the exact candidate and returns structured JSON:

```json
{
  "approve": true,
  "blocking_issues": [],
  "confidence": 0.93
}
```

Rules are strict: approve only if the candidate has no blocking flaws. No politeness inflation.

### Phase 5 — Decide
- **All approve** — return the answer. Done.
- **Any block + rounds remaining** — feed objections back, revise the candidate, re-vote.
- **Max rounds or time exhausted** — fallback.

### Fallback Behavior

When unanimity fails, the behavior is deterministic:

- **Strict mode on + deadlock:** Return the best candidate (highest voter confidence) with unresolved objections attached. `consensus_status: "deadlock"`
- **Strict mode on + one blocker:** `consensus_status: "near-consensus"`, answer still returned.
- **Strict mode off:** Majority-supported candidate returned as success. `consensus_status: "near-consensus"`

Malformed vote output (model didn't return valid JSON) counts as a **block**, not an abstention. This prevents silent failures from inflating apparent unanimity.

---

## Configuration

Everything lives in `config.yaml`. No code edits needed for normal use.

### Server

```yaml
server:
  host: 127.0.0.1
  port: 8000
  log_level: info      # debug | info | warning | error
```

### Agents

Swap models, add agents, or change roles freely:

```yaml
agents:
  - name: analyst
    model: openai/gpt-4o
    role: Critical Analyst
  - name: skeptic
    model: anthropic/claude-sonnet-4-5
    role: Skeptical Reviewer
  - name: optimizer
    model: xai/grok-4
    role: Creative Optimizer
```

Model names use [LiteLLM format](https://docs.litellm.ai/docs/providers) — prefix with the provider (`openai/`, `anthropic/`, `xai/`, `groq/`, etc.). Any model LiteLLM supports works here.

### Debate

```yaml
debate:
  enabled: true
  max_rounds: 2              # max vote-revise cycles
  strict_unanimity: true     # require all agents to approve
  max_total_seconds: 25      # wall-clock timeout for entire debate
  fallback_on_deadlock: true # return best answer if consensus fails
  malformed_vote_is_block: true
```

### Custom Presets

Define your own presets and map them to model names:

```yaml
presets:
  my_preset:
    debate:
      max_rounds: 3
      strict_unanimity: false

virtual_models:
  presets:
    zettai-mine: my_preset
```

Now `zettai-mine` appears in `/v1/models` and applies your custom settings.

---

## API Reference

### `GET /health`

Returns server status, loaded agents, detected API key names, and config state.

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "agents": [
    {"name": "analyst", "model": "openai/gpt-4o", "role": "Critical Analyst"},
    {"name": "skeptic", "model": "anthropic/claude-sonnet-4-5", "role": "Skeptical Reviewer"},
    {"name": "optimizer", "model": "xai/grok-4", "role": "Creative Optimizer"}
  ],
  "detected_api_keys": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
  "strict_unanimity": true,
  "output_mode": "ghost",
  "startup_time": 1776160427.65
}
```

### `GET /v1/models`

Returns an OpenAI-compatible model list with all virtual model names.

### `POST /v1/chat/completions`

Standard OpenAI chat completions endpoint. Supports both streaming and non-streaming responses.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zettai-ittchi",
    "messages": [{"role": "user", "content": "Explain recursion simply."}],
    "stream": false
  }'
```

The endpoint tolerates all standard OpenAI fields (`tools`, `tool_choice`, `response_format`, `temperature`, `max_tokens`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`, `user`, `n`) — unknown fields are accepted silently, never rejected.

---

## Project Structure

```
zettai-ittchi/
  .env.example        API key template
  .env                Your actual keys (git-ignored)
  config.yaml         All runtime configuration
  requirements.txt    Python dependencies
  start_zettai.bat    One-click Windows launcher
  zettai_ittchi/
    __init__.py       Version string
    app.py            FastAPI routes and middleware
    config.py         YAML + env loading, preset resolution
    schemas.py        Pydantic models (OpenAI-compat + internal)
    providers.py      LiteLLM async wrapper, parallel fan-out
    prompts.py        All debate prompt templates
    debate.py         Debate orchestrator (draft/critique/synthesize/vote)
    consensus.py      Vote parsing, unanimity check, fallback logic
    stream.py         OpenAI-compatible SSE streaming
    utils.py          Logging, request IDs, timestamps
  logs/               Runtime logs (auto-created)
```

---

## Requirements

- Python 3.11+
- API keys for at least one supported LLM provider
- Works on Windows, macOS, and Linux

---

## What This Is Not

Zettai Ittchi is deliberately minimal. v1 does **not** include:

- A database or persistent state
- Multi-turn memory beyond the incoming `messages` array
- Tool-call execution or Cursor Agent mode passthrough
- A browser dashboard
- Automatic prompt-type routing
- An install wizard

It is one local server, one config file, one run command, one Cursor endpoint.

---

## License

MIT
