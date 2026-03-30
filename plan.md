# Plan — Telephony Agent: Bugs & Improvements

**Date:** 2026-03-30
**Branch:** `main`

---

## Status Legend

- 🔴 **CRITICAL** — Causes silent failures, crashes, or incorrect data
- 🟠 **MAJOR** — Meaningful malfunction or code quality issue that affects reliability
- 🟡 **MINOR** — Cosmetic, DX, or low-risk issue
- 💡 **IMPROVEMENT** — Not a bug; architectural enhancement or feature addition

---

## Immediate Bugs (Fix Before Next Deploy)

### 🔴 1. Double webhook — `participant_disconnected` + `_monitor_farewell` both report outcome

**File:** `agent.py` — `entrypoint()`, lines 636–643 and 659–686

Both of these fire for a normal call ending:
- `on_participant_disconnected` fires when the SIP participant leaves → calls `report_outcome("COMPLETED")`
- `_monitor_farewell()` detects a farewell phrase → calls `report_outcome("COMPLETED")` then `hangup_call()`

When the farewell monitor triggers the hangup, it deletes the room, which disconnects the participant, which fires `on_participant_disconnected`, which fires the webhook a second time. The API backend has no idempotency guard on the webhook endpoint — this creates duplicate `CallAttempt` records and can corrupt the `CallSchedule` state.

**Fix plan:**
- Add a module-level `_outcome_reported` flag (or asyncio `Event`) per call session
- In `report_outcome()`, set the flag on first call and return early if already set
- OR remove the `participant_disconnected` webhook and rely solely on the farewell monitor + explicit `end_call` tool invocations
- Add idempotency on the API side (see `code-review.md` Phase 5)

---

### 🔴 2. `aiohttp` missing from `requirements.txt`

**File:** `requirements.txt`

`aiohttp` is used in `report_outcome()` to POST the webhook, but is not listed as a dependency. On a fresh install (or in the Docker build), this import will succeed only if `aiohttp` is already installed as a transitive dependency of another package. If that transitive dependency is removed or the import chain changes, webhook delivery silently breaks.

**Fix plan:**
```
# Add to requirements.txt:
aiohttp>=3.9.0
```

---

### 🔴 3. `SarvamTTS` class is dead code with a broken import

**File:** `agent.py`, lines 50–71

The `SarvamTTS` class references `SarvamAI` on line 57:
```python
self.client = SarvamAI(api_subscription_key=api_key)
```
`SarvamAI` is never imported anywhere in the file. If this class were ever instantiated, it would raise a `NameError`. The class is never instantiated — it's fully dead code. The actual Sarvam TTS is configured via `_build_tts()` using the `sarvam.TTS` plugin.

**Fix plan:** Delete the entire `SarvamTTS` class (lines 50–71). It is not used and actively misleads anyone reading the code.

---

### 🟠 4. Greeting hardcoded as "priya" — contradicts system prompt persona "Manisha"

**File:** `agent.py`, line 720

```python
instructions="The candidate has answered. Greet them with exactly: Hello, this is priya calling from Bhanzu..."
```

The system prompt for the default (Bhanzu) mode defines the persona as "Manisha". The override instruction at line 720 hardcodes "priya" (lowercase). This means the agent's system prompt says "you are Manisha" but then the forced greeting makes it say "I'm priya". This is confusing and breaks persona consistency.

**Fix plan:**
- Change "priya" → "Manisha" to match the system prompt
- Or better: remove the forced greeting override and let the system prompt handle the opening (it already has explicit opening instructions)

---

### 🟠 5. `asyncio.ensure_future` is deprecated — use `asyncio.create_task`

**File:** `agent.py`, line 692

```python
asyncio.ensure_future(_monitor_farewell())
```

`asyncio.ensure_future` is deprecated in Python 3.10+ and removed in 3.13. The correct replacement is `asyncio.create_task()`, which also gives you a reference to cancel the task later.

**Fix plan:**
```python
farewell_task = asyncio.create_task(_monitor_farewell())
```
Store the task reference so it can be cancelled if the session ends without the monitor detecting a farewell.

---

### 🟠 6. No cancellation path for `_monitor_farewell` background task

**File:** `agent.py`, `_monitor_farewell()` function

The farewell monitor runs indefinitely until it detects a phrase or encounters an unhandled exception. If the session ends normally (e.g., connection drops, room deleted externally), the task continues polling and will error silently forever. In Python, uncancelled background tasks can keep the event loop alive longer than expected.

**Fix plan:**
- Store the task handle: `farewell_task = asyncio.create_task(_monitor_farewell())`
- Cancel it in the `participant_disconnected` handler: `farewell_task.cancel()`
- Add `except asyncio.CancelledError: pass` inside `_monitor_farewell` to handle clean cancellation

---

### 🟠 7. Room name collision risk — only 9000 possible values

**File:** `make_call.py`, line 36 and `agent.py` (room naming in API's `jobs.js`)

```python
room_name = f"call-{phone_number.replace('+', '')}-{random.randint(1000, 9999)}"
```

With 9000 possible suffixes, simultaneous calls to the same number will likely collide room names, causing the second dispatch to fail or join the wrong room.

**Fix plan:**
```python
import uuid
room_name = f"call-{phone_number.replace('+', '')}-{uuid.uuid4().hex[:8]}"
```

---

### 🟠 8. `BACKEND_WEBHOOK_URL` defaults to `localhost` with no production warning

**File:** `agent.py`, line 32

```python
BACKEND_WEBHOOK_URL = os.getenv("BACKEND_WEBHOOK_URL", "http://localhost:4000/api/call-screening/webhook/call-outcome")
```

In production on Railway, if this env var is not set, all webhook reports silently go to `localhost:4000` (which doesn't exist) and the outcomes are lost. The agent will log a generic `Webhook failed` error but nothing indicates the root cause.

**Fix plan:**
```python
BACKEND_WEBHOOK_URL = os.getenv("BACKEND_WEBHOOK_URL")
if not BACKEND_WEBHOOK_URL:
    logger.warning("BACKEND_WEBHOOK_URL not set — call outcomes will NOT be reported!")
    BACKEND_WEBHOOK_URL = "http://localhost:4000/api/call-screening/webhook/call-outcome"
```

---

### 🟠 9. `ctx.shutdown()` called on outbound call failure — may not exist

**File:** `agent.py`, line 728

```python
except Exception as e:
    logger.error(f"Failed to place outbound call: {e}")
    if schedule_id:
        await report_outcome(schedule_id, "NO_ANSWER")
    ctx.shutdown()   # ← unclear if this method exists on JobContext
```

The LiveKit Agents SDK `JobContext` does not consistently expose a `shutdown()` method across versions. If this method doesn't exist, the exception handler itself throws, masking the original error. The proper cleanup on outbound failure is to let the entrypoint return normally — the SDK handles cleanup.

**Fix plan:** Remove `ctx.shutdown()`. Let the function return after reporting the outcome. The SDK will clean up the room and worker slot automatically.

---

### 🟡 10. Dead imports — `cartesia`, `livekit-plugins-deepgram`, `fastapi`, `uvicorn`

**Files:** `agent.py` line 11, `requirements.txt`

- `cartesia` is imported in `agent.py` but never used
- `livekit-plugins-deepgram` is in `requirements.txt` but the code uses OpenAI STT (not Deepgram)
- `fastapi` and `uvicorn` are in `requirements.txt` but there is no web server in this agent

These add to Docker image size and import time for no benefit. They also mislead developers into thinking Deepgram/Cartesia/FastAPI are part of the system.

**Fix plan:**
- Remove `from livekit.plugins import cartesia` from the import list in `agent.py`
- Remove `livekit-plugins-deepgram`, `fastapi`, `uvicorn[standard]` from `requirements.txt`

---

### 🟡 11. `.env` vs `.env.local` inconsistency between code and README

**Files:** `agent.py:21`, `make_call.py:10`, `setup_trunk.py:7` vs `README.md`

All three Python files call `load_dotenv(".env")`. The README instructs users to create `.env.local`. This means users following the README will have a working local file but the code will load nothing (or will load a separate `.env` if it exists).

**Fix plan:** Either:
- Change README to say `cp .env.example .env` (simpler, matches code)
- Or change code to `load_dotenv(".env.local")` to match README

---

### 🟡 12. `gpt-5-nano` model name may be incorrect

**File:** `agent.py`, line 611

```python
llm=openai.LLM(model="gpt-5-nano"),
```

As of 2026-03-30, `gpt-5-nano` is not a documented OpenAI model tier. If this model does not exist, every call will fail during LLM initialization with an API error. Verify this is the correct model identifier in OpenAI's current API. Standard small models are `gpt-4o-mini`, `o4-mini`, etc.

**Fix plan:** Verify model name against OpenAI's current model list. Add it to an env var so it can be updated without code changes:
```python
llm=openai.LLM(model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")),
```

---

### 🟡 13. Hardcoded Bhanzu default prompt is client-specific, not a system default

**File:** `agent.py`, lines 327–528

The "else" branch in `OutboundAssistant.__init__` that activates when no custom prompt is provided uses a very specific "Manisha / Bhanzu BDA Screening" script. This is a client/test prompt that has leaked into the codebase as the system default. Any test call dispatched without a prompt (e.g., from `make_call.py`) will present as a Bhanzu recruiter.

**Fix plan:**
- Keep it for now (lowest priority)
- Long-term: replace with a generic "AI Recruiter" fallback prompt, or require `prompt` to always be present in dispatch metadata (and fail loudly if missing)

---

## Improvements (Schedule Separately)

### 💡 1. Webhook retry logic

If the POST to `BACKEND_WEBHOOK_URL` fails (network error, API down, timeout), the outcome is silently lost and the `CallSchedule` stays in `IN_PROGRESS` forever. Add exponential backoff retry (3 attempts, 2s/4s/8s delays) inside `report_outcome()`.

---

### 💡 2. Enable noise cancellation for telephony quality

**File:** `agent.py`, lines 14 and 630

```python
# noise_cancellation,  # ← commented out in imports
# noise_cancellation=noise_cancellation.BVCTelephony(),  # ← commented out in session
```

`BVCTelephony` is specifically tuned for phone call audio (8kHz, SIP codec artifacts). Disabling it reduces speech recognition quality on noisy calls. The package is already in `requirements.txt` as `livekit-plugins-noise-cancellation`.

**Fix plan:** Uncomment both lines. Test with a real call to verify compatibility with current Silero VAD + OpenAI STT pipeline.

---

### 💡 3. Configurable model selection via environment variables

Currently STT model, LLM model, TTS model, voice, and language are all hardcoded. Add env vars so they can be tuned per deployment without code changes:

```python
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "simran")
SARVAM_TTS_LANGUAGE = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")
```

---

### 💡 4. Replace polling farewell monitor with event-driven detection

The `_monitor_farewell()` function polls session history every 1.5 seconds. This approach:
- Has up to 1.5s lag between farewell detection and hangup
- Creates unnecessary CPU wake-ups during long calls
- Is fragile to changes in the LiveKit `session.history.messages` API

A better approach is to subscribe to the session's `on_message_added` event (if LiveKit Agents SDK supports it) or hook into the `end_call` tool as the primary hangup path, making farewell monitoring a true last-resort fallback.

---

### 💡 5. Separate prompt templates from `agent.py`

The Bhanzu default prompt (200+ lines) is embedded directly in the agent's constructor. This makes it hard to update prompts without touching agent logic. Move templates to:
- A `prompts/` directory with separate `.txt` or `.md` files
- Loaded at startup with `open("prompts/bhanzu_bda.txt").read()`
- This also enables per-deployment prompt overrides via volume mounts in Docker

---

### 💡 6. Pin dependency versions in `requirements.txt`

Current `requirements.txt` uses `>=` bounds, which means any future breaking release of `livekit-agents`, `livekit-plugins-*`, or `openai` will silently break the agent on next build. Pin to exact versions after validating a working install:

```
livekit-agents==0.12.x
livekit-plugins-openai==0.x.x
livekit-plugins-sarvam==0.x.x
livekit-plugins-silero==0.x.x
```

---

### 💡 7. Add call outcome metrics / observability

Currently outcomes are only visible via API database queries. Add structured logging to make it easy to monitor:
- Call success rate (COMPLETED vs NO_ANSWER)
- Average call duration
- Webhook delivery failure rate
- Model latency (STT / LLM / TTS)

LiveKit Agents SDK exposes timing metadata on the session — log these at call end.

---

### 💡 8. Multi-language support

The TTS is set to `en-IN` (Indian English). Sarvam supports multiple Indian languages (`hi-IN`, `ta-IN`, `te-IN`, etc.). Add a `language` field to the dispatch metadata so recruiters can select the call language per batch.

---

### 💡 9. Inbound call support

The agent currently has a stub for inbound calls (line 730: "No phone number in metadata — treating as inbound/web call"). This could be a full inbound screening flow for candidates who call in directly. Worth building if that use case arises.

---

## Prioritised Fix Order

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Fix double webhook (issue #1) | Small |
| 2 | Add `aiohttp` to `requirements.txt` (issue #2) | Trivial |
| 3 | Delete dead `SarvamTTS` class (issue #3) | Trivial |
| 4 | Fix `ctx.shutdown()` on call failure (issue #9) | Trivial |
| 5 | Fix greeting "priya" → "Manisha" (issue #4) | Trivial |
| 6 | Add `asyncio.create_task` + cancellation (issues #5, #6) | Small |
| 7 | Add `BACKEND_WEBHOOK_URL` production warning (issue #8) | Trivial |
| 8 | Remove dead imports/deps (issue #10) | Trivial |
| 9 | Fix `.env` vs `.env.local` inconsistency (issue #11) | Trivial |
| 10 | Verify/fix `gpt-5-nano` model name (issue #12) | Small |
| 11 | Fix room name collision with UUID (issue #7) | Small |
| 12 | Add webhook retry logic (improvement #1) | Medium |
| 13 | Enable noise cancellation (improvement #2) | Small |
| 14 | Configurable model env vars (improvement #3) | Small |
| 15 | Replace polling farewell monitor (improvement #4) | Medium |
| 16 | Separate prompt templates (improvement #5) | Medium |
| 17 | Pin dependency versions (improvement #6) | Small |
| 18 | Observability / metrics (improvement #7) | Large |
| 19 | Multi-language support (improvement #8) | Large |
