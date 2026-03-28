import logging
import os
import json
import time
import asyncio
from dotenv import load_dotenv
import aiohttp
from livekit import agents, api, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions, get_job_context, function_tool, RunContext
from livekit.plugins import (
    openai,
    cartesia,
    sarvam,
    # noise_cancellation,  
    silero,
)
from livekit.agents import llm
from typing import Annotated, Optional

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

async def report_outcome(schedule_id: str, outcome: str, duration: int = None):
    if not schedule_id:
        return
    logger.info(f"Reporting {outcome} to webhook for schedule_id {schedule_id}")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"scheduleId": schedule_id, "outcome": outcome}
            if duration is not None:
                payload["durationSec"] = duration
            async with session.post(BACKEND_WEBHOOK_URL, json=payload) as resp:
                logger.info(f"Webhook response: {resp.status}")
    except Exception as e:
        logger.error(f"Webhook failed: {e}")


# Sarvam TTS integration wrapper
class SarvamTTS:
    def on(self, event, callback):
        # Dummy event handler for LiveKit compatibility
        pass

    def __init__(self, api_key, target_language_code="hi-IN"):
        self.client = SarvamAI(api_subscription_key=api_key)
        self.target_language_code = target_language_code
        # LiveKit expects a capabilities attribute with streaming property
        self.capabilities = type("Capabilities", (), {"streaming": False})()
        # LiveKit expects a sample_rate attribute (e.g., 8000 Hz or as per Sarvam API)
        self.sample_rate = 8000  # Set to Sarvam's actual sample rate if known
        # LiveKit expects a num_channels attribute (e.g., 1 for mono audio)
        self.num_channels = 1

    async def synthesize(self, text, conn_options=None, **kwargs):
        response = self.client.text_to_speech.convert(
            text=text,
            target_language_code=self.target_language_code
        )
        # The response may contain audio content or a URL, depending on Sarvam API
        return response

def _build_tts():
    """Configure the Text-to-Speech provider using Sarvam AI Bulbul."""
    logger.info("Using Sarvam TTS: bulbul:v3, voice: simran, lang: en-IN")
    return sarvam.TTS(target_language_code="en-IN", model="bulbul:v3", speaker="simran", speech_sample_rate=8000)



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
    
    # parse metadata sent by the dispatch script (or API server)
    schedule_id = None
    phone_number = None
    candidate_name = "Candidate"
    prompt_role = "Software Engineer"
    resume_text = "Not provided."
    jd_text = "Not provided."
    prompt_text = ""
    total_minutes = 10
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

            # If the prompt is long (>100 chars), treat it as a custom prompt.
            # Short values like "Software Engineer" are just role titles → use default prompt.
            if len(raw_prompt.strip()) > 100:
                prompt_text = raw_prompt
                prompt_role = "Custom"
            else:
                prompt_text = ""
                prompt_role = raw_prompt or "Software Engineer"
    except Exception:
        logger.warning("No valid JSON metadata found. This might be an inbound call.")

    # Initialize function context
    fnc_ctx = TransferFunctions(ctx, phone_number)

    # Initialize the Agent Session with plugins

    session = AgentSession(
        # Use Silero VAD (required for non-streaming STT)
        vad=silero.VAD.load(),
        # Use OpenAI gpt-4o-mini-transcribe for STT
        stt=openai.STT(model="gpt-4o-mini-transcribe", language="en"),
        # Use OpenAI GPT-5-nano for LLM
        llm=openai.LLM(model="gpt-5-nano"),
        # Use Sarvam bulbul:v3 ratan for TTS
        tts=_build_tts(),
        userdata=fnc_ctx,
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
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVCTelephony(), 
            close_on_disconnect=True,
        ),
    )

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Caller/Participant disconnected: {participant.identity}")
        if schedule_id:
            try:
                duration = int(time.time() - agent._call_start_time)
                asyncio.create_task(report_outcome(schedule_id, "COMPLETED", duration))
            except Exception as e:
                logger.error(f"Error sending disconnect webhook: {e}")

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
                messages = session.history.messages
                if len(messages) > checked_count:
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
                                    if schedule_id:
                                        duration = int(time.time() - agent._call_start_time)
                                        await report_outcome(schedule_id, "COMPLETED", duration)
                                    await hangup_call()
                                    return
                    checked_count = len(messages)
            except Exception as e:
                logger.debug(f"Monitor error: {e}")

    # Start the farewell monitor as a background task
    asyncio.ensure_future(_monitor_farewell())

    if phone_number:
        logger.info(f"Initiating outbound SIP call to {phone_number}...")
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
    # The agent name "outbound-caller" is used by the dispatch script to find this worker
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller", 
        )
    )
