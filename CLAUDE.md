# CLAUDE.md — Zariya Telephony Agent

> **IMPORTANT:** Keep this file updated whenever you change model providers, add new prompt modes, modify the webhook contract, or change how the agent dispatches/ends calls. An out-of-date CLAUDE.md is worse than none.

---

## Project Overview

This is the **Python AI voice agent** that powers Zariya's outbound call-screening system. It runs as a long-lived LiveKit worker process, picks up dispatch jobs sent by the Node.js API (`zibs-resume-parser-api`), dials candidates via a Vobiz SIP trunk, conducts AI-powered screening interviews, and reports outcomes back to the API via webhook.

**Part of the Zariya monorepo ecosystem:**
```
zariya/resume/
├── zibs-resume-parser-api/     # Node.js backend — dispatches call jobs via AgentDispatchClient
├── zibs-resume-parser-ui/      # Next.js frontend
└── telephony-agent/            ← YOU ARE HERE (Python LiveKit worker)
```

**How calls flow:**
```
Recruiter (UI)
  → API creates CallBatch + CallSchedule records
  → API enqueues job into node-resque
  → Resque worker: dispatchToAgentEngine() → LiveKit AgentDispatchClient.createDispatch()
  → LiveKit dispatches to THIS agent worker (agent_name: "outbound-caller")
  → Agent dials candidate via Vobiz SIP
  → Candidate answers → AI interview/screening conversation
  → Call ends → agent POSTs outcome to BACKEND_WEBHOOK_URL
  → API updates CallSchedule status, creates CallAttempt record
```

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | **Only branch** — development and production use the same branch |

> This repo has a single `main` branch. It deploys directly to Railway via Dockerfile on push to `main`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Runtime | Python 3.12 |
| Framework | LiveKit Agents SDK (`livekit-agents`) |
| VAD | Silero (`livekit-plugins-silero`) |
| STT | OpenAI `gpt-4o-mini-transcribe` via `livekit-plugins-openai` |
| LLM | OpenAI `gpt-5-nano` via `livekit-plugins-openai` |
| TTS | Sarvam Bulbul `bulbul:v3`, speaker `simran`, lang `en-IN` via `livekit-plugins-sarvam` |
| Telephony | LiveKit SIP + Vobiz SIP trunk (outbound calls) |
| Webhook | `aiohttp` HTTP POST to `zibs-resume-parser-api` |
| Deployment | Railway (Docker, `python:3.12-slim`, `restart: always`) |

---

## File Structure

```
telephony-agent/
├── agent.py           # Main worker — all agent logic lives here
├── make_call.py       # Dev utility — manually dispatch a test call
├── setup_trunk.py     # One-time setup — sync Vobiz credentials to LiveKit SIP trunk
├── Dockerfile         # Docker build — Railway uses this
├── railway.toml       # Railway deployment config (Dockerfile builder, always restart)
├── requirements.txt   # Python dependencies
├── .env.example       # Template for environment variables
├── .gitignore         # Excludes .env, .env.local, __pycache__, .venv
└── transfer_call.md   # Guide for SIP REFER (call transfer) setup and troubleshooting
```

---

## `agent.py` — Architecture

### Key Components

#### `entrypoint(ctx: agents.JobContext)`
The main function LiveKit calls for every dispatched job. Flow:
1. Parse job metadata (scheduleId, phone_number, candidate_name, prompt, resume, jd)
2. Initialize `AgentSession` with VAD + STT + LLM + TTS
3. Create `OutboundAssistant` with the candidate-specific prompt
4. Start session
5. Register `participant_disconnected` event handler → sends webhook on hang-up
6. Start `_monitor_farewell()` background task → detects farewell phrases in agent speech and triggers hangup
7. Dial the candidate via `ctx.api.sip.create_sip_participant()` with `wait_until_answered=True`
8. On answer: trigger greeting via `session.generate_reply()`
9. On failure: call `report_outcome(schedule_id, "NO_ANSWER")`

#### `OutboundAssistant(Agent)`
Builds the full LLM system prompt. Two modes:
- **Custom prompt mode**: When `metadata.prompt` is longer than 100 characters, use it as the full system prompt (call-screening use case — prompt comes from `CallTemplate.promptBody`)
- **Default mode**: Falls back to a hardcoded Bhanzu BDA screening prompt (kept for legacy/test; should not be used for Zariya calls)

Candidate context (name, resume, JD) is injected into whichever prompt is active.

Always appends the call-ending logic block (highest priority rules for detecting and acting on end signals).

#### `TransferFunctions(llm.ToolContext)`
Provides the `transfer_call` LLM tool — allows the AI to transfer the call to another SIP number. Resolves the destination to a full `sip:number@domain` URI using `VOBIZ_SIP_DOMAIN`.

#### `report_outcome(schedule_id, outcome, duration)`
Fires a `POST` to `BACKEND_WEBHOOK_URL` with `{ scheduleId, outcome, durationSec }`. Outcomes:
- `"COMPLETED"` — call ended normally (candidate disconnected or farewell phrase detected)
- `"NO_ANSWER"` — outbound dial failed (exception during `create_sip_participant`)
- `"FAILED"` — (reserved, sent by API's retry logic — not sent directly from agent)

#### `_monitor_farewell()`
Background polling task (1.5s interval). Scans session message history for farewell phrases in agent speech. When detected, waits 2s then calls `report_outcome("COMPLETED")` and `hangup_call()`. This handles cases where the LLM doesn't invoke the `end_call` tool.

#### `hangup_call()`
Deletes the LiveKit room via `ctx.api.room.delete_room()`, which terminates the call for all participants.

---

## Metadata Contract (API → Agent)

The API dispatches jobs via `AgentDispatchClient.createDispatch()` with this metadata JSON:

```json
{
  "scheduleId": "uuid",
  "phone_number": "+91XXXXXXXXXX",
  "candidate_name": "Jane Doe",
  "prompt": "...full prompt text from CallTemplate.promptBody (>100 chars)...",
  "jd": "Job Title: ...\nCompany: ...",
  "resume": "...parsed resume JSON string...",
  "total_minutes": 10,
  "templateQuestions": []
}
```

**Prompt routing logic:** If `len(prompt.strip()) > 100` → use as custom prompt. Otherwise treat as a role title and fall back to the Bhanzu default. For Zariya's call-screening use case, `prompt` is always the full `CallTemplate.promptBody` (always > 100 chars).

---

## Webhook Contract (Agent → API)

After each call, the agent POSTs to `BACKEND_WEBHOOK_URL`:

```http
POST /api/call-screening/webhook/call-outcome
Content-Type: application/json

{
  "scheduleId": "uuid",
  "outcome": "COMPLETED" | "NO_ANSWER",
  "durationSec": 142
}
```

The API uses this to update `CallSchedule.status` and create/update `CallAttempt` records.

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `LIVEKIT_URL` | ✅ | LiveKit project WebSocket URL |
| `LIVEKIT_API_KEY` | ✅ | LiveKit API key |
| `LIVEKIT_API_SECRET` | ✅ | LiveKit API secret |
| `OPENAI_API_KEY` | ✅ | OpenAI API key (STT + LLM) |
| `SARVAM_API_KEY` | ✅ | Sarvam API key (TTS) |
| `OUTBOUND_TRUNK_ID` | ✅ | LiveKit SIP trunk ID (starts with `ST_...`) |
| `VOBIZ_SIP_DOMAIN` | ✅ | Vobiz SIP domain (e.g., `xxx.sip.vobiz.ai`) — used for call transfers |
| `BACKEND_WEBHOOK_URL` | ✅ | Full URL for the API webhook endpoint (default: `http://localhost:4000/api/call-screening/webhook/call-outcome`) |
| `DEFAULT_TRANSFER_NUMBER` | Optional | Phone number for default call transfers |
| `VOBIZ_USERNAME` | Setup only | Vobiz SIP username (used by `setup_trunk.py`) |
| `VOBIZ_PASSWORD` | Setup only | Vobiz SIP password (used by `setup_trunk.py`) |
| `VOBIZ_OUTBOUND_NUMBER` | Setup only | Your DID (outbound caller ID) (used by `setup_trunk.py`) |

> ⚠️ `BACKEND_WEBHOOK_URL` defaults to `localhost:4000` if not set. In production on Railway, this MUST be set to the deployed API URL or all outcomes will be silently lost.

---

## Running Locally

```bash
cd telephony-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies (or use uv)
pip install -r requirements.txt

# Copy and fill env
cp .env.example .env
# Edit .env with your keys

# One-time: sync Vobiz credentials to LiveKit trunk
python setup_trunk.py

# Start the agent worker (stays running, listens for dispatch jobs)
python agent.py start

# In a separate terminal: manually trigger a test call
python make_call.py --to +91XXXXXXXXXX
```

---

## Deployment (Railway)

- Railway builds using `Dockerfile` (Python 3.12 slim)
- CMD: `python agent.py start`
- Restart policy: `ALWAYS`, max 5 retries
- All environment variables must be set in the Railway project settings
- The agent runs as a persistent worker — it does not serve HTTP; it connects to LiveKit and listens

---

## Known Issues (as of 2026-03-30)

See `plan.md` for the full list. Critical items:

1. 🔴 `SarvamTTS` class (lines 50–71) is dead code — references `SarvamAI` which is not imported; would crash if instantiated
2. 🔴 Double webhook: both `participant_disconnected` AND `_monitor_farewell` can fire `report_outcome` for the same call — duplicate reports sent to API
3. 🔴 `aiohttp` missing from `requirements.txt` — the webhook call will fail on a fresh install
4. 🟠 Greeting hardcoded as "priya" (line 720) but system prompt persona is "Manisha"
5. 🟠 `asyncio.ensure_future` is deprecated — use `asyncio.create_task`
6. 🟡 Room name collision risk: `random.randint(1000, 9999)` gives only 9000 values

---

## Important Notes for AI Implementations

- This is a **Python** codebase — no JavaScript, no TypeScript
- The LiveKit Agents SDK is the central framework — all session/room/dispatch operations go through it
- The agent runs as a **single-file worker** (`agent.py`) — there is intentionally no web server, no REST API, no database
- System prompts are built **at call time** from metadata received in the dispatch job — they are not static
- The `end_call` tool and the farewell monitor are TWO separate hangup mechanisms that must stay in sync
- When changing the LLM model, also update `gpt-5-nano` → new model in `entrypoint()` and this CLAUDE.md
- When changing TTS provider/voice, update `_build_tts()` and the comment in `entrypoint()` and this CLAUDE.md
