# Telephony Agent

The outbound voice-calling worker for the Resume Parser platform. This repo runs a long-lived LiveKit agent that receives dispatch jobs from the backend, places outbound calls over the SIP trunk, conducts AI-driven screening conversations, and reports outcomes back to the API.

For overall platform onboarding, start with [`../README.md`](../README.md).

## What This Repo Does

- Connects to LiveKit as a worker named `outbound-caller`.
- Waits for dispatch jobs created by the backend call-screening system.
- Dials candidates through the configured outbound SIP trunk.
- Builds a candidate-specific AI prompt using metadata from the dispatch payload.
- Runs the screening conversation with speech recognition, LLM reasoning, and speech synthesis.
- Supports transfer-to-human behavior when the prompt or tools request it.
- Reports `COMPLETED` or `NO_ANSWER` outcomes to the backend webhook.

## Where This Fits In The System

```text
Frontend
  -> Backend creates CallBatch and CallSchedule rows
  -> Backend Resque worker dispatches LiveKit job
  -> This repo receives the dispatch
  -> This repo places the outbound call
  -> This repo POSTs call outcome back to backend webhook
```

## Tech Stack

| Area | Choice |
| --- | --- |
| Runtime | Python 3.12 |
| Container runtime | Docker |
| Agent framework | LiveKit Agents SDK |
| VAD | Silero |
| STT | OpenAI via LiveKit plugin |
| LLM | OpenAI via LiveKit plugin |
| TTS | OpenAI TTS |
| Telephony | LiveKit SIP + Vobiz trunk |
| Webhook client | `aiohttp` |

## File Guide

- `agent.py`: the main worker and almost all runtime behavior.
- `make_call.py`: local utility to manually dispatch a test call.
- `setup_trunk.py`: helper for updating the outbound SIP trunk details in LiveKit.
- `Dockerfile`: container image used in deployment and local dev.
- `docker-compose.yml`: recommended local development entrypoint for the worker.
- `.env.example`: environment variable template.
- `railway.toml`: Railway deployment configuration.
- `transfer_call.md`: notes for transfer behavior and SIP transfer troubleshooting.
- `plan.md`: implementation notes and known issues backlog.

## `agent.py` Responsibilities

`agent.py` is the core of the repo. It handles:

- Loading environment variables from `.env`.
- Starting the LiveKit worker process.
- Receiving job metadata from the backend.
- Creating the conversation session with speech and language providers.
- Managing outbound dialing.
- Detecting hang-up or farewell conditions.
- Posting final call outcomes back to the backend.
- Exposing the `transfer_call` tool to the LLM.

Important metadata fields expected from the backend include:

- `scheduleId`
- `phone_number`
- `candidate_name`
- `prompt`
- `resume`
- `jd`
- `total_minutes`
- `templateQuestions`

## Expected Contract With The Backend

The backend dispatches this agent through LiveKit and expects a webhook back to:

`/api/call-screening/webhook/call-outcome`

Webhook payload shape:

```json
{
  "scheduleId": "uuid",
  "outcome": "COMPLETED",
  "durationSec": 142
}
```

If the worker cannot reach the backend webhook, call schedules in the backend will not close out correctly.

## Environment Variables

Core variables:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`
- `OUTBOUND_TRUNK_ID`
- `VOBIZ_SIP_DOMAIN`
- `BACKEND_WEBHOOK_URL`

Common optional/supporting variables:

- `OPENAI_TTS_MODEL`
- `OPENAI_TTS_VOICE`
- `OPENAI_TTS_INSTRUCTIONS`
- `DEFAULT_TRANSFER_NUMBER`
- `VOBIZ_USERNAME`
- `VOBIZ_PASSWORD`
- `VOBIZ_OUTBOUND_NUMBER`

Use `.env` in this repo, not `.env.local`.

## Local Development Policy

This repo should run in Docker on developer machines. Do not rely on ad hoc host-level Python setups for normal team development.

### Start the worker

```bash
docker compose up --build
```

### Stop the worker

```bash
docker compose down
```

The compose file mounts the repo into the container and loads variables from `.env`.

## Local Development Notes

- If the backend runs on your host machine, set `BACKEND_WEBHOOK_URL` to `http://host.docker.internal:4000/api/call-screening/webhook/call-outcome`.
- Keep the backend running before testing the agent.
- Keep a LiveKit project and SIP trunk configured for the credentials in your `.env`.
- `make_call.py` is for direct agent dispatch testing and is separate from the normal backend-driven flow.

## Manual Test Call

With the worker already running:

```bash
docker compose exec telephony-agent python make_call.py --to +919999999999
```

This is only a quick connectivity test. Normal product testing should go through the backend and frontend call-screening workflow.

## Deployment

Production deployment is container-based:

- Railway builds from `Dockerfile`.
- The process is a long-lived worker, not an HTTP API server.
- Environment variables must be set in Railway for the worker to function.

## Troubleshooting

- Worker starts but never receives jobs: check LiveKit URL/key/secret and confirm the backend dispatches `agent_name="outbound-caller"`.
- Calls dispatch but webhook state never updates: check `BACKEND_WEBHOOK_URL` from inside the container.
- SIP dialing fails: check `OUTBOUND_TRUNK_ID`, `VOBIZ_*` values, and trunk configuration.
- Transfer behavior fails: check `DEFAULT_TRANSFER_NUMBER` and `VOBIZ_SIP_DOMAIN`.
- No speech or bad speech behavior: verify OpenAI credentials and model-related envs.
