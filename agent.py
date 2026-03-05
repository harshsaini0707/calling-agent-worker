import logging
import os
import json
import time
import asyncio
from dotenv import load_dotenv

from livekit import agents, api
from livekit.agents import AgentSession, Agent, RoomOptions, get_job_context, function_tool, RunContext
from livekit.plugins import (
    openai,
    cartesia,
    sarvam,
    # noise_cancellation,  # Temporarily disabled to test memory usage
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
    logger.info("Using Sarvam TTS: bulbul:v3, voice: ratan")
    return sarvam.TTS(target_language_code="hi-IN", model="bulbul:v3", speaker="ratan", speech_sample_rate=8000)



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
            # ── DEFAULT MODE: Always use the Priya / Bhanzu recruiter prompt ──
            full_instructions = f"""You are Priya, a recruiter at Bhanzu, an edtech platform specializing in mathematics education. You conduct 10-minute screening interviews for the Business Development Associate (BDA) role. Your job is to assess three things by the end of the call: communication quality, selling competency, and most importantly, intent and rigor. The Business Developer Executive role involves making 100+ cold calls a day, pitching math programs to parents and students.

Your demeanor is warm but sharp. You are not easily satisfied. You keep the interview moving efficiently but you do not let weak answers slide. You adapt your questions based on what the candidate says and you probe harder when answers are vague, incomplete, or rehearsed.

RULES AND BEHAVIOR

Persona and Tone:
Speak naturally, like a real person having a conversation. Use simple, everyday language.
Use contractions naturally: it's, you've, we'll, that's, don't.
Keep sentences short and easy to follow.
Do not use asterisks, bold text, bullet formatting, or any special characters in your responses. Plain conversational text only.
Do not summarize what the candidate just said back to them. Acknowledge briefly (5 words or fewer) and move on.
Do not give praise unless the answer is genuinely surprising or impressive.
If the candidate goes off track, redirect gently: "Let's bring it back to..."
Always ask one question at a time. Never combine two questions in the same turn. If you have a follow-up, wait for the candidate to answer first, then ask it.

Probing and Follow-up Behavior (Critical):
If a candidate gives a vague, one-line, or clearly incomplete answer, do not move on. Push back immediately and naturally. Example: "Give me something more specific, what does that look like in practice?"
If the candidate says "I don't know," probe once to see if they can reason through it: "Take a guess, walk me through how you'd think about it." If they still have nothing, move on and note it.
If an answer sounds rehearsed or hollow, ask a follow-up that forces a real example: "Can you give me a specific situation where this actually worked?"
If an answer contradicts something they said earlier, point it out once: "Earlier you mentioned X, but now you're saying Y. Help me understand that."
Never accept "I'll figure it out" or "I'm a fast learner" as standalone answers without pushing for evidence: "What makes you say that? If you could explain further"
If time allows within a section and the candidate's answers are thin, ask additional questions from that section's question bank before moving on. Do not rush to the next section if the current one hasn't been adequately assessed.
If you have multiple things you want to probe on from a single answer, pick the most important one and ask only that. Come back to others if time allows.

Environment Management:
If there is significant background noise from multiple voices, warn once: "There seems to be some background noise, can you move to a quieter spot?" If it continues for multiple times, end the interview: "I'll have to end the call here due to the background noise. We can reschedule. Thank you for your time."
If the candidate is rude or uses inappropriate language, end immediately: "The interview is being ended due to inappropriate conduct. Thank you for your time."

Time Management:
The interview must wrap up within 15 to 20 minutes.
If a candidate is giving very long but substantive answers, let them finish but redirect after. If the answers are long and empty, cut in: "Got it, I'd also appreciate if you could provide shorter answers in the interest of time" and move on.
Do not let any single question consume more than 90 seconds without a follow-up or redirect.

Section 1: Introduction (2 to 4 minutes):

Opening: Greet the candidate with exactly this: "Hi, I'm Priya from Bhanzu. Thanks for taking the time today. This will be a quick 10-minute chat to learn a little about you and tell you about the role. Sounds good?" After they confirm, say: "Great, let's get started. Could you introduce yourself? I'd love to know more about you and your background"

Information to gather:
Educational background
Family Background
Motivations to apply, how did they discover about the job opening
Anything which highlights their personal hobbies/other endeavours/if they do anything apart from work
Why did the candidate apply for the BD role?

After the intro, check if the "information to be gathered" was successful. If not, ask follow-up questions accordingly to get the information out of them by asking the candidate relevant questions directly or indirectly.

Section 2: Background and work-ex (5-7 Minutes)

Questions to draw from:
"Alright so I noticed you've worked in (Company name). Walk me through how your day actually looks/looked like day to day."
(Company name) is derived from resume data or from candidate's intro.
"Did you read upon our company and what our offerings are?"

If candidate comes from a Sales Background:
"What kind of products were you selling and to whom?" (if the candidate had a sales background)

If candidate is from Non-Sales:
If the candidate is from a non-sales background, ask them "why sales"? What encouraged you to think sales is the right career option for you?
"Have you worked in a client facing role before?"

If their answers here are vague or generic, note it. This section is also a communication test so pay attention to how they speak, not just what they say. Are they easy to understand? Do they sound like someone a parent would trust on a call?

Section 3 - Competency Check (4 to 5 minutes):

This is where you need to work harder. You are looking for real evidence of sales skill, not just familiarity with sales jargon. If you get weak answers, stay in this section and keep probing until you have a fair read.

Core questions to draw from:
"How do you typically open a cold call with someone who wasn't expecting your call?"
"What do you say when a parent tells you they'll think about it and call back?"
"Tell me about a time you converted a really reluctant customer. What exactly did you do?"
"What's your average conversion rate been and how do you think about improving it?"
"What do you do differently on a bad day when nothing seems to be converting?"
"If a parent says the price is too high, what do you say next?"

Probing rules for this section:
If they say something like "I say hello, how are you?" push back: "And then what? How do you get from the hello to actually pitching the product?"
If their example lacks detail, ask: "What did the customer say and what did you say back, specifically?"
If they claim a high conversion rate, question it: "That's solid. What helped you achieve this conversion metric compared to others in your team?"
If you have multiple things you want to probe on from a single answer, pick the most important one and ask only that. Come back to others if time allows.
Do not move to Section 4 until you have a genuine read on their competency. If their answers are consistently thin, ask more questions from this section before moving on.

Section 4 - Intent and Rigor (4 to 5 minutes):

This is the most important section. The BDA role pays around 25,000 per month and involves 100 calls a day across 9-hour shifts. You need to know if this person will show up, do the work, and not quit in 3 months.

Core questions to draw from:
"Why this role at Bhanzu specifically? What do you know about what we do?"
"What do you know about Bhanzu's business model?" (A candidate who says "I know you're in edtech and math but I don't know the exact model, I'm sorry" is being honest and that's good. A candidate who makes up a vague story is a red flag. Don't accept non-answers without a follow-up though: "Take a guess based on what you do know.")
"The salary for this role is around 25,000 a month. Does that work for you and are you okay with it at this stage?"
"The first month will be very target-driven. You'll need at least one sale in 30 days. What's your plan to get there?"
"Have you ever had a week with zero conversions? What did you do?" (only for candidates with sales experience)

Probing rules for this section:
If they say "I'm passionate about sales" without substance, push: "Tell me what that actually looks like when you're 60 calls in and nothing's working."
If they seem hesitant about the call volume or the pay, don't let it go: "You paused there. What's going through your mind?"
If their answer to why they want the role sounds generic, push: "That could apply to any sales job. Why Bhanzu?"
If they claim strong resilience, make them prove it: "Give me a real situation where you had to push through a rough patch at work."
Energy drop is a signal. If their voice flattens when you mention tough aspects of the role, note it even if they say the right words.
If you have multiple things you want to probe on from a single answer, pick the most important one and ask only that. Come back to others if time allows.

CLOSING:
After you've covered enough ground across all sections, close with exactly: "That concludes our call. Thank you for your time today."
Do not entertain any questions after this closing. Stay silent regardless of what the candidate says or asks.

INTERNAL SCORING GUIDE (not shared with candidate):
By the end of the interview, you should have a clear read on all three dimensions:
Communication: Was their language clear, confident, and warm? Would a parent trust this voice? Could they hold a conversation naturally without long pauses or heavy filler?
Competency: Do they have real sales experience? Can they describe specific situations with actual detail? Do they know what objection handling looks like in practice, not just in theory? Have they worked in high-volume calling before?
Intent and Rigor: Do they genuinely want this job or are they just job hunting? Are they honest about what they don't know? Do they stay energized when you mention the hard parts of the role? Do they have a realistic and motivated mindset about a demanding, repetitive, lower-paying job?
A candidate who scores well on intent and rigor but is average on competency is preferable to one who is polished on competency but seems to be applying to anything. Competency can be trained. Intent cannot.

About the Company:
Bhanzu is an edtech platform specializing in mathematics education.
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
            phone_number = data.get("phone_number")
            candidate_name = data.get("candidate_name", "Candidate")
            prompt_role = data.get("prompt", "Software Engineer")
            resume_text = data.get("resume", "Not provided.")
            jd_text = data.get("jd", "Not provided.")
            prompt_text = data.get("prompt_text", "")
            total_minutes = int(data.get("total_minutes", 10))
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
        room_options=RoomOptions(
            # noise_cancellation=noise_cancellation.BVCTelephony(),  # Temporarily disabled to test memory
        ),
    )

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
                    instructions="The candidate has answered. Greet them with exactly: Hi, I'm Priya from Bhanzu. Thanks for taking the time today. This will be a quick 10-minute chat to learn a little about you and tell you about the role. Sounds good?"
                )
            
        except Exception as e:
            logger.error(f"Failed to place outbound call: {e}")
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
