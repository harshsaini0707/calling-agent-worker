import logging
import os
import json
import time
import asyncio
import hmac
import hashlib
from dotenv import load_dotenv
import aiohttp
from livekit import agents, api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    EndpointingOptions,
    InterruptionOptions,
    JobProcess,
    RoomInputOptions,
    RunContext,
    TurnHandlingOptions,
    function_tool,
    get_job_context,
)
from livekit.plugins import (
    openai,
    deepgram,
    sarvam,
    anthropic,
    silero,
)
from livekit.agents import llm
from typing import Annotated, Optional

try:
    from livekit.agents import inference
except ImportError:
    inference = None

# Telephony-grade noise cancellation. Kept optional so local dev without the
# plugin installed still imports; BVCTelephony is enabled when present (see entrypoint).
try:
    from livekit.plugins import noise_cancellation
except ImportError:
    noise_cancellation = None

try:
    from livekit.plugins import assemblyai
except ImportError:
    assemblyai = None

try:
    from livekit.plugins import cartesia
except ImportError:
    cartesia = None

try:
    from livekit.plugins import elevenlabs
except ImportError:
    elevenlabs = None

# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outbound-agent")


# TRUNK ID - This needs to be set after you crate your trunk
# You can find this by running 'python setup_trunk.py --list' or checking LiveKit Dashboard
OUTBOUND_TRUNK_ID = os.getenv("OUTBOUND_TRUNK_ID")
SIP_DOMAIN = os.getenv("VOBIZ_SIP_DOMAIN") 
BACKEND_WEBHOOK_URL = os.getenv("BACKEND_WEBHOOK_URL", "http://localhost:4000/api/call-screening/webhook/call-outcome")
CALL_SCREENING_WEBHOOK_SECRET = os.getenv("CALL_SCREENING_WEBHOOK_SECRET", "").strip()
DEFAULT_STT_PROVIDER = os.getenv("DEFAULT_STT_PROVIDER", "openai").strip() or "openai"
DEFAULT_OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"
DEFAULT_DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3").strip() or "nova-3"
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "claude-haiku-4-5").strip() or "claude-haiku-4-5"
DEFAULT_TTS_PROVIDER = os.getenv("DEFAULT_TTS_PROVIDER", "sarvam").strip() or "sarvam"
DEFAULT_TTS_VOICE_ID = os.getenv("SARVAM_TTS_SPEAKER", "simran").strip() or "simran"
DEFAULT_TTS_LANGUAGE = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN").strip() or "en-IN"
DEFAULT_TTS_PACE = os.getenv("SARVAM_TTS_PACE", "0.95").strip() or "0.95"
DEFAULT_AI_CONFIG = {
    "version": 2,
    "stt": {
        "provider": DEFAULT_STT_PROVIDER,
        "model": DEFAULT_OPENAI_STT_MODEL if DEFAULT_STT_PROVIDER == "openai" else DEFAULT_DEEPGRAM_STT_MODEL,
        "language": "en-IN" if DEFAULT_STT_PROVIDER == "deepgram" else "en",
        "options": {},
    },
    "llm": {
        "provider": "openai" if DEFAULT_LLM_MODEL.startswith("gpt-") else "anthropic",
        "model": DEFAULT_LLM_MODEL,
        "temperature": 0.4,
        "options": {},
    },
    "tts": {
        "provider": DEFAULT_TTS_PROVIDER,
        "model": "bulbul",
        "voice": DEFAULT_TTS_VOICE_ID,
        "language": DEFAULT_TTS_LANGUAGE,
        "pace": 0.95,
        "options": {},
    },
}

REQUIRED_RUNTIME_ENV_KEYS = [
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "OPENAI_API_KEY",
    "SARVAM_API_KEY",
    "OUTBOUND_TRUNK_ID",
    "BACKEND_WEBHOOK_URL",
]


def get_missing_runtime_env_keys():
    return [key for key in REQUIRED_RUNTIME_ENV_KEYS if not str(os.getenv(key, "")).strip()]


def validate_runtime_env():
    missing_keys = get_missing_runtime_env_keys()
    if missing_keys:
        logger.error(
            "Telephony agent runtime configuration is incomplete. Missing: %s",
            ", ".join(missing_keys),
        )
        return False

    if BACKEND_WEBHOOK_URL.startswith("http://localhost"):
        logger.error(
            "BACKEND_WEBHOOK_URL=%s is not reachable from Docker local mode. Use host.docker.internal or a remote URL.",
            BACKEND_WEBHOOK_URL,
        )
        return False

    return True

def _get_messages(session):
    """Safely extract messages from session depending on LiveKit version."""
    try:
        if hasattr(session, 'history'):
            if hasattr(session.history, 'messages'):
                return session.history.messages() if callable(session.history.messages) else session.history.messages
            elif callable(session.history):
                h = session.history()
                if hasattr(h, 'messages'):
                    return h.messages() if callable(h.messages) else h.messages
        elif hasattr(session, 'chat_ctx') and hasattr(session.chat_ctx, 'messages'):
            return session.chat_ctx.messages() if callable(session.chat_ctx.messages) else session.chat_ctx.messages
    except Exception as e:
        logger.debug(f"Error getting messages: {e}")
    return []

def _collect_transcript(session) -> list:
    """Extract full conversation as [{role, text}] from session history."""
    transcript = []
    try:
        messages = _get_messages(session)
        for msg in messages:
            text = ""
            if hasattr(msg, "content") and isinstance(msg.content, str):
                text = msg.content
            elif hasattr(msg, "text_content"):
                text = msg.text_content
            else:
                text = str(msg)
            text = text.strip()
            if text:
                role = getattr(msg, "role", "unknown")
                # Normalise LiveKit role names to agent/user
                if role == "assistant":
                    role = "agent"
                transcript.append({"role": role, "text": text})
    except Exception as e:
        logger.warning(f"Could not collect transcript: {e}")
    return transcript


async def report_outcome(
    schedule_id: str,
    outcome: str,
    duration: int = None,
    call_uuid: str = None,
    transcript: list = None,
):
    if not schedule_id:
        return
    logger.info(f"Reporting {outcome} to webhook for schedule_id {schedule_id}")
    try:
        async with aiohttp.ClientSession() as http_session:
            payload = {"scheduleId": schedule_id, "outcome": outcome}
            if duration is not None:
                payload["durationSec"] = duration
            if call_uuid:
                payload["callUuid"] = call_uuid
            if transcript:
                payload["transcript"] = transcript

            # Serialize payload to match EXACT bytes sent over the wire
            payload_bytes = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
            
            timestamp_sec = int(time.time())
            headers = {
                "Content-Type": "application/json",
                "X-Call-Screening-Timestamp": str(timestamp_sec),
            }

            if CALL_SCREENING_WEBHOOK_SECRET:
                msg = f"{timestamp_sec}.".encode('utf-8') + payload_bytes
                signature = hmac.new(
                    CALL_SCREENING_WEBHOOK_SECRET.encode('utf-8'),
                    msg,
                    hashlib.sha256
                ).hexdigest()
                headers["X-Call-Screening-Signature"] = f"sha256={signature}"
            else:
                logger.warning("CALL_SCREENING_WEBHOOK_SECRET is not configured; webhook auth will fail against hardened backend")

            async with http_session.post(BACKEND_WEBHOOK_URL, data=payload_bytes, headers=headers) as resp:
                response_body = await resp.text()
                logger.info(f"Webhook response: {resp.status} — {response_body[:200]}")
    except Exception as e:
        logger.error(f"Webhook failed: {e}")


def get_bulbul_model(speaker: str) -> str:
    v2_speakers = {"anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"}
    return "bulbul:v2" if speaker in v2_speakers else "bulbul:v3-beta"


def _coerce_tts_pace(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid SARVAM_TTS_PACE=%s. Falling back to 0.95.", value)
        return 0.95


DEFAULT_AI_CONFIG["tts"]["pace"] = _coerce_tts_pace(DEFAULT_TTS_PACE)


def _normalize_string(value, fallback: str) -> str:
    normalized = str(value or "").strip()
    return normalized or fallback


def _legacy_ai_config(metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    stt_model = _normalize_string(metadata.get("sttModel"), DEFAULT_STT_PROVIDER).lower()
    llm_model = _normalize_string(metadata.get("llmModel"), DEFAULT_LLM_MODEL)
    tts_provider = _normalize_string(metadata.get("ttsProvider"), DEFAULT_TTS_PROVIDER).lower()
    tts_voice = _normalize_string(metadata.get("ttsVoiceId") or metadata.get("speaker"), DEFAULT_TTS_VOICE_ID)

    stt_provider = "deepgram" if stt_model == "deepgram" else "openai"
    llm_provider = "openai" if llm_model.startswith("gpt-") else "anthropic"
    if llm_model.startswith("gemini"):
        llm_provider = "google"

    tts_default_models = {
        "sarvam": "bulbul",
        "elevenlabs": "eleven_turbo_v2_5",
        "cartesia": "sonic-3",
        "deepgram": "aura-2",
        "livekit-inference": "cartesia/sonic-3",
    }

    return {
        "version": 2,
        "stt": {
            **DEFAULT_AI_CONFIG["stt"],
            "provider": stt_provider,
            "model": DEFAULT_DEEPGRAM_STT_MODEL if stt_provider == "deepgram" else DEFAULT_OPENAI_STT_MODEL,
            "language": "en-IN" if stt_provider == "deepgram" else "en",
        },
        "llm": {
            **DEFAULT_AI_CONFIG["llm"],
            "provider": llm_provider,
            "model": llm_model,
        },
        "tts": {
            **DEFAULT_AI_CONFIG["tts"],
            "provider": tts_provider,
            "model": tts_default_models.get(tts_provider, "bulbul"),
            "voice": tts_voice,
        },
    }


def _normalize_ai_config(metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    source = metadata.get("aiConfig")
    if not isinstance(source, dict):
        return _legacy_ai_config(metadata)

    return {
        "version": 2,
        "stt": {**DEFAULT_AI_CONFIG["stt"], **(source.get("stt") or {})},
        "llm": {**DEFAULT_AI_CONFIG["llm"], **(source.get("llm") or {})},
        "tts": {**DEFAULT_AI_CONFIG["tts"], **(source.get("tts") or {})},
    }


def _inference_available(kind: str) -> bool:
    if inference is not None:
        return True
    logger.warning("LiveKit Inference %s requested but this livekit-agents version has no inference module.", kind)
    return False


def _build_stt(ai_config: dict):
    config = ai_config.get("stt", {})
    provider = _normalize_string(config.get("provider"), DEFAULT_AI_CONFIG["stt"]["provider"]).lower()
    model = _normalize_string(config.get("model"), DEFAULT_AI_CONFIG["stt"]["model"])
    language = _normalize_string(config.get("language"), DEFAULT_AI_CONFIG["stt"]["language"])
    options = config.get("options") if isinstance(config.get("options"), dict) else {}

    if provider == "livekit-inference" and _inference_available("STT"):
        logger.info("Using LiveKit Inference STT: model=%s language=%s", model, language)
        return inference.STT(model=model, language=language, extra_kwargs=options)

    if provider == "deepgram":
        if os.getenv("DEEPGRAM_API_KEY"):
            logger.info("Using Deepgram STT: model=%s language=%s", model, language)
            return deepgram.STT(model=model, language=language)
        logger.warning("Deepgram STT requested but DEEPGRAM_API_KEY is not configured; falling back to OpenAI")

    if provider == "assemblyai" and assemblyai is not None:
        logger.info("Using AssemblyAI STT: model=%s language=%s", model, language)
        return assemblyai.STT(model=model, language=language)

    if provider == "sarvam" and hasattr(sarvam, "STT"):
        logger.info("Using Sarvam STT: model=%s language=%s", model, language)
        return sarvam.STT(model=model, language=language)

    if provider not in ("openai", "deepgram"):
        logger.warning("Unsupported or unavailable STT provider requested: %s. Falling back to OpenAI.", provider)

    logger.info("Using OpenAI STT: model=%s language=%s", DEFAULT_OPENAI_STT_MODEL, "en")
    return openai.STT(model=DEFAULT_OPENAI_STT_MODEL, language="en")


def _build_llm(ai_config: dict):
    config = ai_config.get("llm", {})
    provider = _normalize_string(config.get("provider"), DEFAULT_AI_CONFIG["llm"]["provider"]).lower()
    model = _normalize_string(config.get("model"), DEFAULT_LLM_MODEL)
    temperature = config.get("temperature", DEFAULT_AI_CONFIG["llm"]["temperature"])
    options = config.get("options") if isinstance(config.get("options"), dict) else {}

    if provider == "livekit-inference" and _inference_available("LLM"):
        extra_kwargs = {"temperature": temperature, **options}
        logger.info("Using LiveKit Inference LLM: model=%s", model)
        return inference.LLM(model=model, extra_kwargs=extra_kwargs)

    if provider == "openai":
        logger.info("Using OpenAI LLM: model=%s", model)
        responses_api = getattr(openai, "responses", None)
        if responses_api and hasattr(responses_api, "LLM"):
            return responses_api.LLM(model=model, temperature=temperature)
        return openai.LLM(model=model, temperature=temperature)

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            logger.info("Using OpenRouter LLM: model=%s", model)
            return openai.LLM(model=model, base_url="https://openrouter.ai/api/v1", api_key=api_key, temperature=temperature)
        logger.warning("OpenRouter LLM requested but OPENROUTER_API_KEY is not configured; falling back to Anthropic")

    if provider == "google":
        logger.warning("Google Gemini LLM requested but this worker does not load the Gemini plugin yet; falling back to Anthropic")

    if provider not in ("anthropic", "openrouter", "google"):
        logger.warning("Unsupported LLM provider requested: %s. Falling back to Anthropic.", provider)

    logger.info("Using Anthropic LLM: model=%s", DEFAULT_LLM_MODEL)
    return anthropic.LLM(model=DEFAULT_LLM_MODEL, temperature=min(float(temperature), 1.0))


def _build_tts(ai_config: dict):
    """Configure the Text-to-Speech provider using Sarvam Bulbul voices."""
    config = ai_config.get("tts", {})
    provider = _normalize_string(config.get("provider"), DEFAULT_TTS_PROVIDER).lower()
    model = _normalize_string(config.get("model"), "bulbul")
    voice = _normalize_string(config.get("voice") or config.get("voiceId"), DEFAULT_TTS_VOICE_ID)
    language_code = _normalize_string(config.get("language"), DEFAULT_TTS_LANGUAGE)
    pace = _coerce_tts_pace(str(config.get("pace", DEFAULT_TTS_PACE)))
    options = config.get("options") if isinstance(config.get("options"), dict) else {}

    if provider == "livekit-inference" and _inference_available("TTS"):
        logger.info("Using LiveKit Inference TTS: model=%s voice=%s language=%s", model, voice, language_code)
        return inference.TTS(model=model, voice=voice, language=language_code, extra_kwargs=options)

    if provider == "elevenlabs" and elevenlabs is not None:
        logger.info("Using ElevenLabs TTS: model=%s voice=%s language=%s", model, voice, language_code)
        return elevenlabs.TTS(model=model, voice_id=voice, language=language_code)

    if provider == "cartesia" and cartesia is not None:
        logger.info("Using Cartesia TTS: model=%s voice=%s language=%s", model, voice, language_code)
        return cartesia.TTS(model=model, voice=voice, language=language_code)

    if provider == "deepgram":
        logger.info("Using Deepgram TTS: model=%s voice=%s", model, voice)
        return deepgram.TTS(model=f"{model}-{voice}" if voice and voice not in model else model)

    if provider != "sarvam":
        logger.warning("Unsupported or unavailable TTS provider requested: %s. Falling back to Sarvam.", provider)

    speaker = voice.strip().lower() or DEFAULT_TTS_VOICE_ID

    model = get_bulbul_model(speaker)
    logger.info(
        "Using Sarvam TTS: model=%s speaker=%s language=%s pace=%s",
        model,
        speaker,
        language_code,
        pace,
    )
    return sarvam.TTS(
        target_language_code=language_code,
        model=model,
        speaker=speaker,
        pace=pace,
        output_audio_codec="mp3",
        speech_sample_rate=80000,
    )



class TransferFunctions(llm.ToolContext):
    def __init__(self, ctx: agents.JobContext, phone_number: str = None):
        super().__init__(tools=[])
        self.ctx = ctx
        self.phone_number = phone_number

    @llm.function_tool(description="Transfer the call to a human support agent or another phone number.")
    async def transfer_call(self, destination: Optional[str] = None):
        """
        Transfer the call.
        """
        if destination is None:
            destination = os.getenv("DEFAULT_TRANSFER_NUMBER")
            if not destination:
                 return "Error: No default transfer number configured."
        if "@" not in destination:
            # If no domain is provided, append the SIP domain
            if SIP_DOMAIN:
                # Ensure clean number (strip tel: or sip: prefix if present but no domain)
                clean_dest = destination.replace("tel:", "").replace("sip:", "")
                destination = f"sip:{clean_dest}@{SIP_DOMAIN}"
            else:
                # Fallback to tel URI if no domain configured
                if not destination.startswith("tel:") and not destination.startswith("sip:"):
                     destination = f"tel:{destination}"
        elif not destination.startswith("sip:"):
             destination = f"sip:{destination}"
        
        logger.info(f"Transferring call to {destination}")
        
        # Determine the participant identity
        # For outbound calls initiated by this agent, the participant identity is typically "sip_<phone_number>"
        # For inbound, we might need to find the remote participant.
        participant_identity = None
        
        # If we stored the phone number from metadata, we can construct the identity
        if self.phone_number:
            participant_identity = f"sip_{self.phone_number}"
        else:
            # Try to find a participant that is NOT the agent
            for p in self.ctx.room.remote_participants.values():
                participant_identity = p.identity
                break
        
        if not participant_identity:
            logger.error("Could not determine participant identity for transfer")
            return "Failed to transfer: could not identify the caller."

        try:
            logger.info(f"Transferring participant {participant_identity} to {destination}")
            await self.ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self.ctx.room.name,
                    participant_identity=participant_identity,
                    transfer_to=destination,
                    play_dialtone=False
                )
            )
            return "Transfer initiated successfully."
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return f"Error executing transfer: {e}"


class OutboundAssistant(Agent):

    """
    An AI interviewer agent for conducting structured voice interviews.
    """
    def __init__(self, name: str = "Candidate", role: str = "Software Engineer",
                 resume_section: str = "Not provided.", jd_section: str = "Not provided.",
                 prompt_text: str = "", total_minutes: int = 10) -> None:
        wrap_up_minutes = max(total_minutes - 2, 1)

        # Determine if user provided real inputs or this is a default test call
        has_resume = resume_section and resume_section.strip().lower() not in ("not provided.", "not provided", "")
        has_jd = jd_section and jd_section.strip().lower() not in ("not provided.", "not provided", "")
        has_prompt = prompt_text and prompt_text.strip() != ""

        self._call_start_time = time.time()

        # ── CALL-ENDING LOGIC (shared across ALL prompt modes) ──
        end_call_logic = f"""

## CALL ENDING — HIGHEST PRIORITY (OVERRIDES ALL OTHER INSTRUCTIONS)

If the candidate shows ANY intent to stop the conversation, the call must end immediately.

This rule overrides the interview flow, greeting flow, and all other instructions.

- NEVER call end_call before you have greeted the candidate and they have responded at least once. Silence or background noise is NOT a reason to end the call.
- When the candidate explicitly wants to end the call, your ENTIRE response must be ONLY: "Thank you, have a great day!" — nothing else before or after. Then IMMEDIATELY call the end_call tool.
- ANY of these signals mean END NOW: "end this call", "hang up", "disconnect", "stop", "bye", "goodbye", "I have to go", "I'm busy", "not interested", "please stop", "cut the call", "I don't want to continue", "let's end this", "that's all", "I'm done", "no thanks", "can we stop", "please disconnect", "call later", "I didn't apply", "I have not applied", "wrong number", "no I haven't applied", or ANY similar phrase in ANY language.
- IMPORTANT: If the candidate says they did NOT apply for the role, that is an END signal. Say "Sorry for the inconvenience, have a great day!" and call end_call immediately.
- NEVER say more than one sentence when ending. NEVER ask "are you sure?". Just say the one line and call end_call.

Never ask any question once an ending signal is detected.

Never continue the interview after an ending signal.

Never delay ending the call.

---

### Ending Detection (Dynamic)

End the call immediately if the candidate says or implies they want to stop.

This includes direct or indirect phrases such as:

- bye
- goodbye
- stop
- end
- end the call
- end the interview
- end the conversation
- cut the call
- cut the conversation
- hang on later
- call later
- wrong time
- not interested
- no thanks
- please stop
- I'm busy
- I have to go
- I'm done
- wrap up
- finish this
- enough
- disconnect
- talk later
- maybe later
- remove my number
- do not call

ALSO end the call if the candidate:

- gives very short disinterested answers
- sounds annoyed
- refuses to continue
- says they cannot talk
- repeatedly avoids questions

Interpret intent broadly. If unsure, end the call.

---

### Required Closing Message (MANDATORY)

Before ending the call, say ONE polite closing message.

The message must:

- Sound natural and human
- Sound like a recruiter
- Maximum 15 words
- No questions
- No confirmations
- No extra sentences

Examples of valid closing messages:

"Thanks for your time. We will connect through email. Goodbye."

"Thank you for your time today. We will follow up by email. Goodbye."

"Appreciate your time. We will share updates through email. Goodbye."

"Thanks for speaking with us. We will reach out by email. Goodbye."

The wording may vary slightly but must keep the same meaning.

---

### Tool Execution (CRITICAL)

Immediately after the closing message:

Call the end_call tool in the SAME response.

Requirements:

- Closing message must appear first
- Tool call must happen immediately after
- No additional text allowed
- No explanations allowed
- No delay allowed

---

### Strict Behavior Rules

- Never continue talking after the closing message.
- Never generate a second message.
- Never ask questions before ending.
- Never confirm the ending.
- Never ignore an ending signal.
- Always end the call within the same response.
"""

        # ── Build resume/JD context to inject into prompts when available ──
        candidate_context = ""
        if has_resume and has_jd:
            candidate_context = f"""

## Candidate Information
- Name: {name}

## Their Resume
{resume_section}

## Job Description for the Role
{jd_section}

Use the resume and JD above to inform your questions. Reference specific companies, roles, projects, and skills from the resume. Ask how their experience maps to the JD requirements."""
        elif has_resume:
            candidate_context = f"""

## Candidate Information
- Name: {name}

## Their Resume
{resume_section}

Use the resume above to inform your questions. Reference specific companies, roles, projects, and skills the candidate has listed."""
        elif has_jd:
            candidate_context = f"""

## Candidate Information
- Name: {name}

## Job Description for the Role
{jd_section}

Use the job description above to inform your questions. Ask about the key skills and responsibilities mentioned in the JD."""

        if has_prompt:
            # ── CUSTOM PROMPT MODE: Use the user-provided prompt as-is ──
            full_instructions = f"""{prompt_text}

{candidate_context}

{end_call_logic}"""

        else:
            # ── DEFAULT MODE: Always use the Manisha / Bhanzu recruiter prompt ──
            full_instructions = f"""SYSTEM PROMPT — BHANZU BDA SCREENING CALL AGENT
## Conduct the whole interview only in English. Never mention or acknowledge that the interview is in English. Keep the accent in English.
## WHO YOU ARE
You are Manisha, a recruiter at Bhanzu, an edtech company specializing in mathematics
education. You are conducting a structured phone screening for the Business Development
Associate role. Your tone is calm, warm, and professional — not casual, not stiff. You sound
like a real person having a focused conversation, not a robot reading from a script.
You speak in plain, everyday language. Use contractions naturally. Keep your sentences
short. Do not use bullet points, asterisks, bold text, or any special characters in your
responses. Everything you say should read like natural spoken dialogue.

## YOUR GOAL
By the end of this call, you need a clear read on three things: whether the candidate can
communicate clearly and confidently, whether they genuinely want to be in sales, and
whether they understand and accept what this specific role demands. This is not a deep
interview. It is a filter. You are checking if they are worth moving forward, not making a final
hiring call.

## CALL DURATION
The call should last between 8 and 12 minutes. Do not rush through questions to fill a
checklist. If a candidate is giving strong answers, you can spend more time in a section. If
their answers are thin, probe before moving on. Do not end the call early just because you've
asked all the questions — and do not drag it out if you have what you need.

## HOW YOU SPEAK — NON-NEGOTIABLE RULES
Always ask one question at a time. Never combine two questions in the same turn. If you
want to follow up on something, wait for their answer first, then ask the follow-up.
Do not summarize what the candidate just said. Acknowledge in five words or fewer and
move forward. "Got it." or "Okay, makes sense." is enough.
Do not give praise unless something genuinely surprises you. Saying "great answer" after
every response makes the screening feel fake and gives the candidate no real signal.
If the candidate's answer is vague or incomplete, you can push back once with something
like "Can you be more specific about that?" or "Give me a quick example." If their follow-up is
still thin but they've made a genuine attempt, accept it, note it internally, and move on. Do
not push a third time. Pressing past the point of willingness does not get you better
information.
If the candidate seems uncomfortable, is repeating themselves, or is clearly trying to move
past a question, do not persist. Take what they've given you, note it, and move to the next
question.
If the candidate says "I don't know," probe once. Ask them to reason through it: "Take a
guess — how would you think about it?" If they still have nothing, move on and make a note
of it.
If the answer contradicts something they said earlier, flag it once: "Earlier you mentioned X,
but now you're saying Y — can you help me understand that?"
Never accept "I'm a fast learner" or "I'll figure it out" as a complete answer. Follow up once:
"What makes you say that? Give me a quick example." If they still can't substantiate it, move
on.
If the candidate is going on too long and the answer is substantive, let them finish and then
redirect. If the answer is long and empty, cut in with: "Got it — I'd appreciate shorter answers
so we can cover everything in time" and move on.
Do not let any single question go beyond 90 seconds without a follow-up or a redirect.

## ENVIRONMENT AND CONDUCT MANAGEMENT
If there is loud, disruptive background noise, warn once: "There seems to be some
background noise on your end — is there a quieter spot you can move to?" If the noise
continues and makes the call difficult, end it: "I'll have to end the call here because of the
background noise. We can reschedule — thank you for your time."
If the candidate is rude, dismissive, or uses inappropriate language at any point, end the call
immediately: "The call is being ended due to conduct. Thank you for your time." Do not
argue or explain further.
If the candidate cannot hear you or the connection is very poor, try once: "Can you hear me
clearly?" If the problem continues, end the call politely and suggest they call back from a
better connection.

## OPENING THE CALL
Begin every call with exactly this: "Hello, this is Manisha calling from Bhanzu. I'm reaching
out regarding your application for the Business Development Associate role. Is this a good
time to talk?"
If they say it is not a good time, say: "No problem at all. When would be a better time to
connect?" Note the preferred time and end the call politely.
If they confirm it's a good time, say: "Perfect. This will be a quick call — about ten minutes.
I'll ask you a few questions to understand your background and we'll also set some
expectations about the role. Sounds good?"
Once they confirm, move into the first question.

## SECTION 1 — INTRODUCTION
Purpose: Get a baseline read on who this person is, how they communicate, and why they
applied.
Ask: "To start off, could you tell me a little about yourself — your background, what you've
been doing recently, and what brought you to apply for this role?"
Listen without interrupting.
After they finish, check mentally whether you now know their educational background, their
most recent work or activity, and what drew them to apply. If the introduction is missing key
information, prioritise asking about their background, education and then motivation to work
and ensure this data is collected. For example: "You mentioned your work background but I
didn't catch — what's your educational qualification?"
Data to be collected (only to be asked if the candidate does not mention any of these details
in their answer):
- Where are you currently based out of?
- What's your educational qualifications?
- When did the candidate complete their education?
- What's their professional background (if any)
Do not ask about all the missing pieces at once. Ask one at a time.

## SECTION 2 — MOTIVATION AND ROLE FIT
Purpose: Understand whether they genuinely want to be in sales and whether they
understand what this role involves.
Ask: "What is it about a career in sales that appeals to you right now?"
If the answer is vague — something like "I'm a people person" or "I like talking to people" —
push back: "A lot of roles involve talking to people. What is it specifically about sales that
you're drawn to?"
If they come from a non-sales background, ask their one specific reason for being drawn to
sales now? The word "one" is doing important work here: it signals to the candidate that a
focused answer is expected, not a life story, and gives Manisha a natural exit once that
reason is given.
For this section, choose only one follow-up based on the most prominent gap in their
answer: either push on vagueness, ask about prior sales experience, or address a non-sales
background. Do not chain these. Once you have a usable read on their motivation, move to
Section 3 regardless of whether the answer was strong.

## SECTION 2.5 — DATA COLLECTION
If the candidate mentions they're currently in a job or internship, enquire about "could you
confirm what your notice period would be" if you end up joining our organization?
What is your current compensation? If they mention their previous CTC or monthly salary,
also confirm from them if it was all fixed or if it had variable components as well.
Confirm the candidate's salary expectations also and let them know that the role currently
pays 25,000 to Rs.30,000 per month along with additional incentives per sale. Ask if they're
fine with the CTC.
Confirm if the candidate has access to a working laptop with stable internet also.

## SECTION 3 — ROLE EXPECTATION CHECK
Purpose: Make sure they understand and genuinely accept what the role involves before
moving forward.
Say: "I want to give you a clear picture of what this role looks like day-to-day so we're on the
same page. This is a high-volume calling role. You'd be reaching out to parents and
students, making upwards of a hundred calls a day, working 8 hour shifts, five days a week.
Does this match what you were expecting?"
Give them room to respond fully. Do not rush past this.
After their response to this follow-up, do not probe further regardless of what they say. Note
their answer as a signal — positive or negative — and move forward to Section 4. This
section is a reality check, not a negotiation. Once you have their honest reaction, you have
what you need.

## SECTION 4 — CANDIDATE QUESTIONS
Purpose: Give the candidate a chance to ask questions, and handle their queries accurately
and consistently.
Say: "Before we wrap up, do you have any questions for me about the role or the process?"
Give them a moment to think. Do not rush past this. If they say they have no questions,
move directly to the close.
If they do have questions, answer them using only the information below. Do not speculate,
do not elaborate beyond what is listed, and do not offer personal opinions. If they ask
something not covered here, say: "I don't have that detail on hand right now — the team will
share more if you move forward in the process."
How to handle specific questions:
If they ask about the interview process or how many rounds there are, say: "This call was a
telephonic screening to get a sense of your background. After this, there will be one to two
more rounds — one will be more of a skills-based assessment and the other would be with
the hiring manager."
If they ask about compensation or salary, say: "For this role, you'd start on a six-month
internship at 15,000 a month. Once you're converted to a full-time role, the fixed component
moves up to 25,000 a month."
If they ask about job location or whether it's remote, say: "This is an in-office role. You'd be
working out of our office in HSR Layout, Bangalore."
The office timings will be 9PM to 6PM from Monday to Friday.
If they ask a follow-up on any of these topics that goes beyond what's covered above, say:
"I'd want to make sure I give you accurate information — the team will be able to answer that
in more detail during the next round."
Do not get drawn into negotiating compensation, debating the location, or speculating on
timelines. Answer once, clearly, and move on.
Once all their questions are addressed, move to the close.

## SECTION 5 — PRACTICAL CONFIRMATION
Purpose: Confirm logistical readiness and close the screening.
Wait for their answer. Then ask: "And are you comfortable with a five-day work week from
the office?"
If they say no to either, note it clearly. Do not disqualify them on the call — just record it as a
flag.
If they say yes to both, move to the close.
If they ask questions about the role, respond with brief, factual, neutral answers. Do not
speculate about hiring timelines. Do not make promises. Do not share your personal opinion
on their chances. If they ask something you don't know, say: "I don't have that detail on hand
— the team will share more if you move to the next round."

## CLOSING THE CALL
End every call with exactly this: "Thank you so much for your time today. Our team will be in
touch with the next steps. Have a great day."
Do not add anything after this. If they ask about timelines or outcomes, say: "I'm not able to
share that at this stage — the team will be in touch. Have a great day." Then end the call.
IMPORTANT: After saying the closing message, you MUST call the end_call tool immediately to hang up.

## WHAT YOU ARE ASSESSING — INTERNAL NOTES
Do not share this with the candidate.
Communication: Does this person speak clearly and confidently? Would a parent on the
other end of a cold call stay on the line with them? Do they ramble, go blank, or struggle to
find words? Or do they sound natural, warm, and easy to follow?
Sales motivation and fit: Is their reason for wanting to be in sales grounded in something
real, a prior experience, a genuine disposition or does it feel like they applied to anything
available? Candidates who are honest about being newer to sales but show genuine
curiosity are better signals than candidates who overstate enthusiasm without substance.
Acceptance of role realities: Did they absorb the details of the role without flinching, or did
their energy drop when you mentioned call volumes and shift lengths? Did they seem to
already know what they were applying for, or did it catch them off guard? A candidate who
hears the expectations and leans in is a much stronger signal than one who goes quiet or
immediately tries to reframe it.
Honesty and self-awareness: Were they straightforward when they didn't know something, or
did they try to bluff? Did they give you real answers or rehearsed ones? A candidate who
says "honestly I haven't done sales before but here's why I want to try it" is more trustworthy
than one who claims to be a natural closer with nothing to back it up.
{candidate_context}

{end_call_logic}"""

        super().__init__(
            instructions=full_instructions
        )

    @function_tool
    async def end_call(self, ctx: RunContext):
        """Called ONLY when the candidate or agent explicitly asks to end or disconnect the call."""
        elapsed = time.time() - self._call_start_time
        if elapsed < 2:
            logger.info(f"end_call blocked: only {elapsed:.1f}s into the call (minimum 2s)")
            return "Cannot end call yet. The interview has just started. Continue with the greeting."
        logger.info("Ending call as requested")
        await hangup_call()


async def hangup_call():
    """Delete the room to end the call for all participants."""
    ctx = get_job_context()
    if ctx is None:
        return
    await ctx.api.room.delete_room(
        api.DeleteRoomRequest(
            room=ctx.room.name,
        )
    )


# ── LiveKit prewarm ───────────────────────────────────────────────────────────
def prewarm(proc: JobProcess) -> None:
    """Load the Silero VAD once per worker process and reuse across jobs.

    On LiveKit Cloud the worker is long-lived and handles many calls, so loading
    the VAD here (instead of per-session) cuts per-call cold-start latency.
    """
    proc.userdata["vad"] = silero.VAD.load()


# ── Agent server ──────────────────────────────────────────────────────────────
# agent_name MUST stay "outbound-caller": the backend/make_call dispatch targets
# this name to place outbound SIP calls. Do not rename.
server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session(agent_name="outbound-caller")
async def entrypoint(ctx: agents.JobContext):
    """
    Main entrypoint for the agent.
    
    For outbound calls:
    1. Checks for 'phone_number' in the job metadata.
    2. Connects to the room.
    3. Initiates the SIP call to the phone number.
    4. Waits for answer before speaking.
    """
    logger.info(f"Connecting to room: {ctx.room.name}")

    if not validate_runtime_env():
        logger.error("Shutting down telephony job because required runtime env validation failed")
        ctx.shutdown()
        return
    
    # parse metadata sent by the dispatch script (or API server)
    schedule_id = None
    phone_number = None
    candidate_name = "Candidate"
    prompt_role = "Software Engineer"
    resume_text = "Not provided."
    jd_text = "Not provided."
    prompt_text = ""
    total_minutes = 10
    ai_config = DEFAULT_AI_CONFIG
    try:
        if ctx.job.metadata:
            data = json.loads(ctx.job.metadata)
            schedule_id = data.get("scheduleId")
            phone_number = data.get("phone_number")
            candidate_name = data.get("candidate_name", "Candidate")
            raw_prompt = data.get("prompt", "Software Engineer")
            resume_text = data.get("resume", "Not provided.")
            jd_text = data.get("jd", "Not provided.")
            total_minutes = int(data.get("total_minutes", 10))
            ai_config = _normalize_ai_config(data)

            normalized_prompt = str(raw_prompt or "").strip()
            if normalized_prompt:
                prompt_text = normalized_prompt
                prompt_role = "Custom"
            else:
                prompt_text = ""
                prompt_role = "Software Engineer"
    except Exception:
        logger.warning("No valid JSON metadata found. This might be an inbound call.")

    # Initialize function context
    fnc_ctx = TransferFunctions(ctx, phone_number)

    # Initialize the Agent Session with plugins

    session = AgentSession(
        # Silero VAD (required for non-streaming STT), prewarmed once per process.
        vad=ctx.proc.userdata["vad"],
        stt=_build_stt(ai_config),
        llm=_build_llm(ai_config),
        tts=_build_tts(ai_config),
        userdata=fnc_ctx,
        turn_handling=TurnHandlingOptions(
            # min_delay 1.0s (up from the 0.5s default) so a thinking-pause mid-answer
            # isn't treated as end-of-turn — the agent was cutting candidates off.
            # max_delay 3.0s is the hard cap before the turn is forced closed.
            # Telephony note: phone audio is noisier/laggier than web; 1.0s is a sane
            # starting point. Tune up if cut-offs persist on real calls.
            endpointing=EndpointingOptions(min_delay=1.0, max_delay=3.0),
            # adaptive interruption distinguishes real interruptions from backchannel
            # ("mm-hm", "okay"); min_duration 0.5s ignores brief blips/line noise.
            interruption=InterruptionOptions(mode="adaptive", min_duration=0.5),
            # Keep preemptive generation for snappy replies.
            preemptive_generation={"enabled": True, "preemptive_tts": True},
        ),
    )

    # Start the session
    agent = OutboundAssistant(
        name=candidate_name,
        role=prompt_role,
        resume_section=resume_text,
        jd_section=jd_text,
        prompt_text=prompt_text,
        total_minutes=total_minutes,
    )
    # BVCTelephony is the phone-tuned noise-cancellation model (narrowband SIP audio).
    # Do NOT use ai_coustics QUAIL here — that targets wideband web audio.
    nc = noise_cancellation.BVCTelephony() if noise_cancellation is not None else None
    if nc is None:
        logger.warning(
            "noise_cancellation plugin unavailable; running without BVCTelephony"
        )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=nc,
            close_on_disconnect=True,
        ),
    )

    # Track whether we already reported the outcome (to avoid duplicate webhooks)
    outcome_reported = False
    sip_call_uuid = None  # Vobiz call_uuid captured from SIP participant attributes

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        """Capture the Vobiz call_uuid from SIP participant attributes as soon as it joins."""
        nonlocal sip_call_uuid
        try:
            attrs = participant.attributes or {}
            identity = participant.identity or ""
            metadata = participant.metadata or ""
            

            # logger.info(f"=== SIP PARTICIPANT CONNECTED ===")
            # logger.info(f"  Identity: {identity}")
            # logger.info(f"  Metadata: {metadata}")
            # logger.info(f"  All attributes ({len(attrs)} keys):")
            for key, value in attrs.items():
                logger.info(f"    {key} = {value}")
            # logger.info(f"====")
            
            # Try known key names for the Vobiz call UUID
            for key in ("sip.callId", "sip.callID", "sip.call_id", 
                         "sip.callUUID", "sip.call_uuid",
                         "sip.trunkCallId", "sip.extra.X-Call-UUID",
                         "sip.extra.X-Call-Id", "callId", "call_uuid"):
                if key in attrs:
                    sip_call_uuid = attrs[key]
                    logger.info(f"Captured Vobiz call_uuid from '{key}': {sip_call_uuid}")
                    break
            
            if not sip_call_uuid:
                logger.warning(f"Could not find call_uuid in any known attribute key")
        except Exception as e:
            logger.warning(f"Could not capture call_uuid: {e}")

    async def _safe_report(outcome: str, dur: int = None):
        """Report outcome exactly once — includes transcript and call_uuid."""
        nonlocal outcome_reported
        if outcome_reported or not schedule_id:
            return
        outcome_reported = True
        try:
            transcript = _collect_transcript(session)
            logger.info(f"Collected transcript with {len(transcript)} turns")
            await report_outcome(
                schedule_id,
                outcome,
                duration=dur,
                call_uuid=sip_call_uuid,
                transcript=transcript,
            )
        except Exception as e:
            logger.error(f"Failed to report outcome: {e}")

    # Helper to end the call gracefully
    async def _end_call():
        """Disconnect all SIP participants and shut down the agent context."""
        try:
            for p in ctx.room.remote_participants.values():
                if p.identity.startswith("sip_"):
                    logger.info(f"Removing SIP participant: {p.identity}")
                    await ctx.api.room.remove_participant(
                        api.RoomParticipantIdentity(room=ctx.room.name, identity=p.identity)
                    )
        except Exception as e:
            logger.warning(f"Error removing participant: {e}")
        ctx.shutdown()

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Caller/Participant disconnected: {participant.identity}")
        if schedule_id and not outcome_reported:
            try:
                duration = int(time.time() - agent._call_start_time)
                asyncio.create_task(_safe_report("COMPLETED", duration))
            except Exception as e:
                logger.error(f"Error sending disconnect webhook: {e}")

    # Safety-net: when the agent session itself shuts down, always report outcome
    @session.on("close")
    def on_session_close():
        logger.info("AgentSession closed — ensuring outcome is reported")
        if not outcome_reported and schedule_id:
            try:
                duration = int(time.time() - agent._call_start_time)
            except Exception:
                duration = 0
            asyncio.create_task(_safe_report("COMPLETED", duration))

    # Auto-hangup: background task that monitors agent speech for farewell phrases
    FAREWELL_PHRASES = [
        "sorry for the inconvenience",
        "have a great day",
        "have a good day",
        "goodbye",
        "thank you, have a great",
        "thank you, have a good",
        "we will follow up",
        "we will reach out",
        "we'll follow up",
        "we'll reach out",
    ]

    async def _monitor_farewell():
        """Poll chat history every 1.5s to detect farewell phrases in agent messages."""
        checked_count = 0
        while True:
            await asyncio.sleep(1.5)
            try:
                messages = _get_messages(session)
                if messages and len(messages) > checked_count:
                    for msg in messages[checked_count:]:
                        if msg.role == "assistant":
                            text = ""
                            # Extract text content from the message
                            if hasattr(msg, 'content') and isinstance(msg.content, str):
                                text = msg.content.lower()
                            elif hasattr(msg, 'text_content'):
                                text = msg.text_content.lower()
                            else:
                                text = str(msg).lower()
                            
                            for phrase in FAREWELL_PHRASES:
                                if phrase in text:
                                    logger.info(f"Farewell detected in agent message: '{phrase}' — hanging up in 2s")
                                    await asyncio.sleep(2)
                                    duration = int(time.time() - agent._call_start_time)
                                    await _safe_report("COMPLETED", duration)
                                    await _end_call()
                                    return
                    checked_count = len(messages)
            except Exception as e:
                logger.debug(f"Monitor error: {e}")

    # Start the farewell monitor as a background task
    asyncio.ensure_future(_monitor_farewell())

    if phone_number:
        logger.info(f"Initiating outbound SIP call to {phone_number}...")
        if not OUTBOUND_TRUNK_ID:
            logger.error("OUTBOUND_TRUNK_ID is missing. Cannot place outbound SIP call.")
            if schedule_id:
                await report_outcome(schedule_id, "FAILED")
            ctx.shutdown()
            return
        try:
            # Create a SIP participant to dial out
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=OUTBOUND_TRUNK_ID,
                    sip_call_to=phone_number,
                    participant_identity=f"sip_{phone_number}",
                    wait_until_answered=True,
                )
            )
            logger.info("Call answered! Agent is now listening.")
            
            # Reset the call start time NOW (after the call is actually answered)
            agent._call_start_time = time.time()
            
            # Use appropriate greeting based on which prompt mode is active
            has_custom_prompt = prompt_text and prompt_text.strip() != ""
            if has_custom_prompt:
                await session.generate_reply(
                    instructions="The candidate has answered. Greet them warmly and begin the interview as described in your instructions."
                )
            else:
                await session.generate_reply(
                    instructions="The candidate has answered. Greet them with exactly: Hello, this is priya calling from Bhanzu. I'm reaching out regarding your application for the Business Development Associate role. Is this a good time to talk?"
                )
            
        except Exception as e:
            logger.error(f"Failed to place outbound call: {e}")
            if schedule_id:
                await report_outcome(schedule_id, "NO_ANSWER")
            # Ensure we clean up if the call fails
            ctx.shutdown()
    else:
        # Fallback for inbound calls (if this agent is used for that)
        logger.info("No phone number in metadata. Treating as inbound/web call.")
        await session.generate_reply(instructions="Greet the user.")


if __name__ == "__main__":
    # agent_name "outbound-caller" is set on the @server.rtc_session decorator above;
    # the dispatch script / backend uses that name to find this worker.
    agents.cli.run_app(server)
