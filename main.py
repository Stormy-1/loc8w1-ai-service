import json
import os
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import typing_extensions as typing
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
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

CLIENT_INTENTS = {"post_job", "search_workers", "check_status", "cancel_job", "general_chat"}
WORKER_INTENTS = {"find_jobs", "accept_job", "decline_job", "update_availability", "submit_quote", "general_chat"}

class IntentParameters(typing.TypedDict):
    service_type: str | None
    raw_spoken_location: str | None
    urgency: str | None
    price: int | None
    eta: str | None

class Intent(typing.TypedDict):
    action: str
    confidence: str
    parameters: IntentParameters

class PipelineResponse(typing.TypedDict):
    translated_text: str
    tts_input: str
    intent: Intent
    clarification_needed: bool
    clarification_prompt: str | None
    detected_language: str

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

gemini = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=GenerationConfig(
        response_mime_type="application/json",
        response_schema=PipelineResponse
    )
)

SYSTEM_PROMPT = """You are the routing brain for a multilingual voice-assistant serving blue-collar workers and clients in India.
Your input is transcribed speech. Your output must be strictly valid JSON.

INPUTS:
ROLE: {role}
SOURCE_LANG: {user_lang}
TARGET_LANG: {target_lang}
CONTEXT: {context}
TRANSCRIPT: "{transcript}"

STRICT RULES:
1. action MUST be exactly one of: [post_job, search_workers, check_status, cancel_job, accept_job, decline_job, update_availability, submit_quote, find_jobs, general_chat].
2. confidence MUST be one of: [high, medium, low]. If low, clarification_needed MUST be true.
3. service_type MUST be exactly one of: [plumber, carpenter, electrician, painter, mechanic]. Map local slang semantically ('mistri' -> mechanic/carpenter). If ambiguous or unmappable, return null and set clarification_needed to true.
4. urgency MUST be one of: [low, medium, high].
5. price MUST be an integer, no currency symbols.
6. detected_language MUST be a valid BCP-47 code.
7. If the spoken intent seems to contradict the user's ROLE, set action to general_chat and clarification_needed to true. Ask politely in SOURCE_LANG.
8. tts_input must spell out numbers and currencies naturally for speech. e.g., 500 -> "paanch sau rupaye" if SOURCE_LANG is hi.
"""

@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        transcription = await groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(audio.filename, audio_bytes, audio.content_type),
            response_format="verbose_json" 
        )
        return {"transcript": transcription.text, "detected_language": transcription.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        response = await gemini.generate_content_async(prompt)
        result = json.loads(response.text) 
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

@app.post("/api/tts")
def text_to_speech(payload: TTSRequest):
    try:
        from google.cloud import texttospeech
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=payload.text)
        voice = texttospeech.VoiceSelectionParams(language_code=payload.lang, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        
        audio_b64_string = base64.b64encode(response.audio_content).decode('utf-8')
        return {"audio_b64": audio_b64_string}
    except Exception as e:
        return {"audio_b64": None, "fallback": True, "error": str(e)}

@app.get("/api/health")
def health():
    return {"status": "ok"}