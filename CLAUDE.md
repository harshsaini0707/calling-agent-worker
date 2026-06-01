# CLAUDE.md — Telephony Agent

Keep this file updated whenever the LiveKit contract, webhook payload, provider stack, or local run model changes.

## Purpose

This repo is the Python LiveKit worker for outbound candidate screening calls.

It receives dispatch jobs from `zibs-resume-parser-api`, places outbound SIP calls, runs the AI conversation, and posts the outcome back to the backend webhook.

## Repo Boundaries

This repo does not own:

- recruiter UI,
- call batch creation,
- candidate/job persistence,
- retry scheduling logic.

Those live in the backend and frontend repos.

This repo does own:

- LiveKit worker startup,
- outbound dialing,
- conversation prompt assembly,
- transfer tool behavior,
- call termination behavior,
- webhook reporting after the call.

## Important Files

- `agent.py`: main worker and call logic.
- `make_call.py`: manual dispatch utility for quick testing.
- `setup_trunk.py`: trunk update helper.
- `Dockerfile`: container image definition.
- `Dockerfile.local`: local-only Docker build for iterative development.
- `docker-compose.local.yml`: required local development path.
- `docker-compose.yml`: compatibility compose file kept alongside the local variant.
- `preflight.py`: local env existence and Docker-safe webhook checks.
- `.env.example`: variable template.

## Provider Stack In Code

- Runtime: Python 3.12
- Agent framework: LiveKit Agents SDK
- VAD: Silero
- STT: OpenAI plugin
- LLM: OpenAI plugin
- TTS: OpenAI TTS

If the code switches providers, update this file and the README immediately.

## Local Development Rule

Run this repo in Docker on dev machines.

Expected workflow:

```bash
docker compose -f docker-compose.local.yml up --build
```

Use `.env`, not `.env.local`.

When the backend runs on the host machine, `BACKEND_WEBHOOK_URL` usually needs to be:

```text
http://host.docker.internal:4000/api/call-screening/webhook/call-outcome
```

## Backend Contract

The backend dispatches jobs to the agent name:

```text
outbound-caller
```

Typical metadata includes:

- `scheduleId`
- `phone_number`
- `candidate_name`
- `prompt`
- `resume`
- `jd`
- `total_minutes`
- `templateQuestions`
- `aiConfigVersion` (`2` for structured model configuration)
- `aiConfig.stt.provider/model/language/options`
- `aiConfig.llm.provider/model/temperature/options`
- `aiConfig.tts.provider/model/voice/language/pace/options`
- `sttModel` (`openai` default, `deepgram` supported when configured)
- `ttsProvider` (`sarvam` default; unsupported values fall back to Sarvam)
- `ttsVoiceId` (Sarvam speaker id, defaults to `simran`)
- `llmModel` (`claude-haiku-4-5` default; `gpt-*` routes to OpenAI)
- `webhook_url` (full URL to POST call outcome back to; falls back to `BACKEND_WEBHOOK_URL` env var when absent)

Prefer `aiConfig` for new dispatches. The flat model fields remain as legacy fallback.

Webhook payload sent back by this repo:

```json
{
  "scheduleId": "uuid",
  "outcome": "COMPLETED" | "NO_ANSWER" | "FAILED" | "VOICEMAIL" | "PREMATURE_DISCONNECT" | "CALL_REJECTED" | "INVALID_NUMBER",
  "durationSec": 142,
  "callUuid": "optional",
  "transcript": [{"role": "agent|user", "text": "..."}],
  "candidateWordCount": 42,
  "transcriptTurnCount": 8,
  "errorMessage": "optional SIP or runtime error detail"
}
```

Outcomes are classified from the transcript when the call disconnects, unless a
specific outcome is forced (e.g. SIP failure before answer). Per-dispatch
`webhook_url` in job metadata takes precedence over `BACKEND_WEBHOOK_URL`.

## Maintenance Rule

Update `README.md`, `.env.example`, and this file together when any of the following change:

- provider env vars,
- webhook path or payload,
- dispatch metadata shape,
- local Docker workflow,
- deployment model,
- transfer or hangup behavior.
