import json
import os
import asyncio
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.environ.get("GROQ_API_KEY") or not os.environ.get("GEMINI_API_KEY"):
    print("WARNING: API Keys missing in .env file!")

groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

# --- New google.genai client ---
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

CLIENT_INTENTS = {"post_job", "search_workers", "check_status", "cancel_job", "general_chat"}
WORKER_INTENTS = {"find_jobs", "accept_job", "decline_job", "update_availability", "submit_quote", "general_chat"}

# --- JSON Schema for structured output ---
PIPELINE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "translated_text": types.Schema(type=types.Type.STRING),
        "tts_input": types.Schema(type=types.Type.STRING),
        "intent": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "action": types.Schema(type=types.Type.STRING),
                "confidence": types.Schema(type=types.Type.STRING),
                "parameters": types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "service_type": types.Schema(type=types.Type.STRING, nullable=True),
                        "raw_spoken_location": types.Schema(type=types.Type.STRING, nullable=True),
                        "urgency": types.Schema(type=types.Type.STRING, nullable=True),
                        "price": types.Schema(type=types.Type.INTEGER, nullable=True),
                        "eta": types.Schema(type=types.Type.STRING, nullable=True),
                    },
                    required=["service_type", "raw_spoken_location", "urgency", "price", "eta"],
                ),
            },
            required=["action", "confidence", "parameters"],
        ),
        "clarification_needed": types.Schema(type=types.Type.BOOLEAN),
        "clarification_prompt": types.Schema(type=types.Type.STRING, nullable=True),
        "detected_language": types.Schema(type=types.Type.STRING),
    },
    required=["translated_text", "tts_input", "intent", "clarification_needed", "clarification_prompt", "detected_language"],
)

class PipelineRequest(BaseModel):
    transcript: str
    role: str
    user_lang: str
    target_lang: str
    last_two_exchanges: list
    active_job_id: str | None

class TTSRequest(BaseModel):
    text: str
    lang: str

SYSTEM_PROMPT = """You are the routing brain for a multilingual voice-assistant serving blue-collar workers and clients in India.
Your input is transcribed speech (often in Hindi, Marathi, or Gujarati). Your output must be strictly valid JSON.

INPUTS:
ROLE: {role}
SOURCE_LANG: {user_lang}
TARGET_LANG: {target_lang}
CONTEXT: {context}
TRANSCRIPT: "{transcript}"

LANGUAGE HANDLING:
- Hindi (hi): Default working language. Understand colloquial terms like 'mistri', 'karigar', 'kaam chahiye'.
- Marathi (mr): Understand terms like 'kaamgaar' (worker), 'nalkawala' (plumber), 'sutar' (carpenter), 'rangari' (painter). City references: Pune, Nashik, Nagpur, Thane.
- Gujarati (gu): Understand terms like 'karigar' (worker), 'nalwala' (plumber), 'sutar' (carpenter), 'rangar' (painter). City references: Ahmedabad, Surat, Vadodara, Rajkot.
- Detect and return the correct BCP-47 code: hi, mr, gu, en, etc.

CRITICAL: DO NOT infer or hallucinate parameters like 'location' or 'service_type' if they are not explicitly spoken in the transcript. If a detail is missing, return null for that parameter and set clarification_needed to true. Ask the user for the missing detail in clarification_prompt using SOURCE_LANG.

STRICT RULES:
1. action MUST be exactly one of: [post_job, search_workers, check_status, cancel_job, accept_job, decline_job, update_availability, submit_quote, find_jobs, general_chat].
2. confidence MUST be one of: [high, medium, low]. If low, clarification_needed MUST be true.
3. service_type MUST be exactly one of: [plumber, carpenter, electrician, painter, mechanic]. Map regional slang semantically (Hindi 'mistri' -> mechanic/carpenter, Marathi 'nalkawala' -> plumber, Gujarati 'nalwala' -> plumber). If ambiguous or unmappable, return null and set clarification_needed to true.
4. urgency MUST be one of: [low, medium, high]. Default to medium if not stated.
5. price MUST be an integer, no currency symbols.
6. detected_language MUST be a valid BCP-47 code (hi, mr, gu, en, etc.).
7. If the spoken intent seems to contradict the user's ROLE, set action to general_chat and clarification_needed to true. Ask politely in SOURCE_LANG.
8. tts_input must spell out numbers and currencies naturally for speech in SOURCE_LANG. e.g., 500 -> "paanch sau rupaye" (hi), "paachshe rupaye" (mr), "panchso rupiya" (gu).
9. translated_text MUST be a faithful translation. DO NOT add details that were not in the original transcript.
"""

# --- Retry helper for Gemini rate limits ---
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = [5, 15, 30]  # generous backoff for free tier
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]  # fallback chain


async def call_gemini_with_retry(prompt: str) -> dict:
    """Calls Gemini with automatic retry on rate-limit / transient errors.
    Falls back to alternate model if primary is rate-limited.
    """
    last_error = None
    for model_name in GEMINI_MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=PIPELINE_SCHEMA,
                    ),
                )
                return json.loads(response.text)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Retry on rate-limit (429) or transient server errors (5xx)
                if "429" in str(e) or "resource_exhausted" in error_str or "500" in str(e) or "503" in str(e):
                    delay = RETRY_DELAY_SECONDS[min(attempt, len(RETRY_DELAY_SECONDS) - 1)]
                    print(f"[Gemini] {model_name} rate limited (attempt {attempt+1}/{MAX_RETRIES}). Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    # Non-retryable error — break to try next model
                    break
        print(f"[Gemini] All retries exhausted for {model_name}. Trying next model...")
    raise last_error


@app.post("/api/tts")
def text_to_speech(payload: TTSRequest):
    """
    Free TTS using gTTS. Zero API keys or billing required.
    """
    try:
        from gtts import gTTS
        import io

        # gTTS uses 2-letter codes (e.g., 'hi' instead of 'hi-IN')
        lang_code = payload.lang.split('-')[0] if '-' in payload.lang else payload.lang

        # Generate audio directly into memory (no saving to disk)
        tts = gTTS(text=payload.text, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)

        # Convert raw bytes to Base64 for the frontend
        audio_b64_string = base64.b64encode(fp.getvalue()).decode('utf-8')

        return {"audio_b64": audio_b64_string}

    except Exception as e:
        return {"audio_b64": None, "fallback": True, "error": str(e)}


@app.post("/api/pipeline")
async def run_pipeline(payload: PipelineRequest):
    context_str = json.dumps(payload.last_two_exchanges)
    prompt = SYSTEM_PROMPT.format(
        role=payload.role,
        user_lang=payload.user_lang,
        target_lang=payload.target_lang,
        context=context_str,
        transcript=payload.transcript
    )

    try:
        result = await call_gemini_with_retry(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    action = result.get("intent", {}).get("action", "general_chat")
    role = payload.role.upper()

    if role == "CLIENT" and action in WORKER_INTENTS - {"general_chat"}:
        result["intent"]["action"] = "general_chat"
        result["clarification_needed"] = True
        result["clarification_prompt"] = "Yeh action aapke liye available nahi hai."

    elif role == "WORKER" and action in CLIENT_INTENTS - {"general_chat"}:
        result["intent"]["action"] = "general_chat"
        result["clarification_needed"] = True
        result["clarification_prompt"] = "Yeh action aapke liye available nahi hai."

    if result["intent"]["action"] == "submit_quote" and not payload.active_job_id:
        result["intent"]["action"] = "general_chat"
        result["clarification_needed"] = True
        result["clarification_prompt"] = "Kaunsi job ke liye quote dena chahte hain?"

    return result


@app.post("/api/stt")
async def process_audio(audio: UploadFile = File(...)):
    """
    Receives WebM audio from the browser, sends it to Groq Whisper,
    and returns the transcribed text.
    """
    try:
        audio_bytes = await audio.read()

        # Groq requires a tuple with a filename to process the raw bytes
        transcription = await groq_client.audio.transcriptions.create(
            file=("audio.webm", audio_bytes),
            model="whisper-large-v3",
            response_format="text",
            temperature=0.0,
            prompt="Hindi, Marathi, Gujarati. Plumber, carpenter, electrician, mechanic, Pune, Mumbai."
        )

        return {"transcript": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq STT error: {str(e)}")


@app.get("/api/health")
def health():
    return {"status": "ok"}


class WorkerForMatch(BaseModel):
    worker_id: str
    name: str
    distance_km: float
    rating: float
    is_available_now: bool
    is_verified: bool
    jobs_completed: int

class MatchRequest(BaseModel):
    urgency: str
    workers: list[WorkerForMatch]

@app.post("/api/match_workers")
def match_workers(payload: MatchRequest):
    """
    Advanced heuristic matching engine implementing the weighted formula:
    0.35(Distance) + 0.25(Rating) + 0.20(Availability) + 0.20(Trust) + UrgencyBoost
    """
    scored_workers = []

    for worker in payload.workers:
        distance_score = max(0.0, 100.0 - (worker.distance_km * 5.0))
        rating_score = (worker.rating / 5.0) * 100.0
        availability_score = 100.0 if worker.is_available_now else 0.0
        experience_points = min(50.0, worker.jobs_completed * 2.0)
        trust_score = (50.0 if worker.is_verified else 0.0) + experience_points

        base_score = (0.35 * distance_score) + \
                     (0.25 * rating_score) + \
                     (0.20 * availability_score) + \
                     (0.20 * trust_score)

        urgency_boost = 0.0
        if payload.urgency.lower() == "high" and worker.is_available_now:
            urgency_boost = 15.0

        final_score = base_score + urgency_boost

        scored_workers.append({
            "worker_id": worker.worker_id,
            "name": worker.name,
            "match_score": round(final_score, 2),
            "breakdown": {
                "distance": round(distance_score, 1),
                "rating": round(rating_score, 1),
                "trust": round(trust_score, 1)
            }
        })

    scored_workers.sort(key=lambda x: x["match_score"], reverse=True)

    return {"ranked_workers": scored_workers}


# =============================================================================
# VOICE SESSION — Multi-Turn Job Posting Conversation Engine
# -----------------------------------------------------------------------------
# This is a standalone, self-contained feature designed to be merged into the
# main KaamSetu app. It manages a stateful voice dialogue to collect all job
# posting parameters through natural speech. State travels on the CLIENT —
# no server-side session storage is needed.
# =============================================================================

REQUIRED_JOB_PARAMS = ["job_type", "location", "date", "time", "budget"]

class VoiceSessionRequest(BaseModel):
    """
    Single turn in the voice job-posting conversation.
    - transcript: what the user just said (from STT)
    - collected_params: dict of params gathered so far (maintained by frontend)
    - language: BCP-47 code (hi, mr, gu, en)
    - is_confirmation: True when the AI has asked "confirm job?" and user replied
    """
    transcript: str
    collected_params: dict
    language: str = "hi"
    is_confirmation: bool = False


VOICE_SESSION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        # Newly extracted params from this single turn
        "extracted_params": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "job_type":    types.Schema(type=types.Type.STRING, nullable=True),
                "location":    types.Schema(type=types.Type.STRING, nullable=True),
                "date":        types.Schema(type=types.Type.STRING, nullable=True),
                "time":        types.Schema(type=types.Type.STRING, nullable=True),
                "budget":      types.Schema(type=types.Type.STRING, nullable=True),
                "description": types.Schema(type=types.Type.STRING, nullable=True),
            },
            required=["job_type", "location", "date", "time", "budget", "description"],
        ),
        # What the AI should say next (fed directly to /api/tts)
        "next_question": types.Schema(type=types.Type.STRING),
        # True when ALL 5 core params are filled (job_type, location, date, time, budget)
        "is_complete": types.Schema(type=types.Type.BOOLEAN),
        # True ONLY when is_confirmation=True and user said yes/haan/ha
        "job_confirmed": types.Schema(type=types.Type.BOOLEAN),
    },
    required=["extracted_params", "next_question", "is_complete", "job_confirmed"],
)


VOICE_SESSION_PROMPT = """You are a smart multilingual voice assistant for KaamSetu, an Indian gig-labour marketplace.
Your job is to collect all required information to post a job, one natural conversational turn at a time.
Respond ONLY in the user's language: {language}. Be concise and conversational — this is spoken audio.

ALREADY COLLECTED PARAMETERS (do NOT ask about these again):
{collected_params}

WHAT THE USER JUST SAID:
"{transcript}"

IS_CONFIRMATION MODE: {is_confirmation}

YOUR TASK:
1. Extract any NEW job parameters from the user's speech. Map regional terms:
   - job_type: plumber/nalkawala/nalwala, carpenter/sutar, electrician, painter/rangari/rangar, mechanic/mistri
   - location: any city, area, or phrase like "mere paas", "yahan", "mumbai mein"
   - date: aaj=today, kal=tomorrow, parso=day after, or specific dates
   - time: subah=morning, dopahar=afternoon, shaam=evening, raat=night, or specific times
   - budget: any number in rupees. "paanch sau"=500, "hazaar"=1000
   - description: optional details about the problem (pipe leak, wiring issue, etc.)

2. Determine what is STILL MISSING from the 5 required params: job_type, location, date, time, budget.

3. Set next_question to the SINGLE most important missing parameter question in {language}.
   Ask only ONE thing at a time. Do not repeat what's already been answered.
   Use natural spoken language, not formal text.

4. Set is_complete=true ONLY when job_type, location, date, time, AND budget are all known (from collected + newly extracted).

5. If is_complete=true, set next_question to the confirmation question (e.g. "Theek hai! [job_type] ke liye [location] mein [date] ko [time] baje, budget [budget] rupaye. Kya main yeh job post karun?")

6. IF IS_CONFIRMATION MODE IS TRUE:
   - If the user said yes/haan/ha/bilkul/confirm → set job_confirmed=true, next_question="Job post ho gayi! Jald hi koi karigar aapko contact karega."
   - If the user said no/nahi → set job_confirmed=false, next_question="Theek hai, kya aap kuch change karna chahte hain?"

Return strictly valid JSON only."""


@app.post("/api/voice_session")
async def voice_session(payload: VoiceSessionRequest):
    """
    Multi-turn voice job posting conversation engine.
    
    Each call represents ONE turn in the dialogue:
    - Frontend sends the STT transcript + all params collected so far
    - Backend extracts new params and returns the next question for TTS
    - Frontend plays TTS audio, records user reply, and calls this endpoint again
    - Loop continues until is_complete=True, then is_confirmation=True, then job_confirmed=True
    
    Design: Stateless on server side. All state is carried in collected_params by the frontend.
    This makes it easy to integrate into any client (React, mobile app, etc.)
    """
    prompt = VOICE_SESSION_PROMPT.format(
        language=payload.language,
        collected_params=json.dumps(payload.collected_params, ensure_ascii=False),
        transcript=payload.transcript,
        is_confirmation=payload.is_confirmation,
    )

    try:
        last_error = None
        result = None
        for model_name in GEMINI_MODELS:
            for attempt in range(MAX_RETRIES):
                try:
                    response = await gemini_client.aio.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            response_mime_type="application/json",
                            response_schema=VOICE_SESSION_SCHEMA,
                        ),
                    )
                    result = json.loads(response.text)
                    break  # success
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    if "429" in str(e) or "resource_exhausted" in error_str or "500" in str(e) or "503" in str(e):
                        delay = RETRY_DELAY_SECONDS[min(attempt, len(RETRY_DELAY_SECONDS) - 1)]
                        print(f"[VoiceSession] {model_name} rate limited (attempt {attempt+1}/{MAX_RETRIES}). Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        break  # non-retryable, try next model
            if result is not None:
                break
        if result is None:
            raise last_error
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini voice session error: {str(e)}")

    # Merge newly extracted params into the running collection
    # Only update fields that were null/missing before (never overwrite confirmed data)
    new_params = result.get("extracted_params", {})
    merged_params = dict(payload.collected_params)
    for key, value in new_params.items():
        if value is not None and key not in merged_params:
            merged_params[key] = value

    # Recheck completeness on the server side (guard against Gemini mistakes)
    server_is_complete = all(merged_params.get(p) for p in REQUIRED_JOB_PARAMS)

    return {
        "collected_params": merged_params,
        "next_question": result.get("next_question", ""),
        "is_complete": server_is_complete,
        "job_confirmed": result.get("job_confirmed", False),
    }