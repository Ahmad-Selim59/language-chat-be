import os
import httpx
import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import asyncio

from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GCP_SA_PATH = os.path.join(ROOT_DIR, "gcp-service-account.json")

GCP_CREDENTIALS = None

try:
    if not os.path.exists(GCP_SA_PATH):
        print(f"WARNING: GCP service account JSON not found at {GCP_SA_PATH}. STT/TTS will fail.")
    else:
        GCP_CREDENTIALS = service_account.Credentials.from_service_account_file(
            GCP_SA_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
except Exception as e:
    print(f"ERROR: Failed to load GCP credentials: {e}")

def get_gcp_credentials():
    if not GCP_CREDENTIALS:
        raise RuntimeError("GCP credentials not initialized.")
    
    if not GCP_CREDENTIALS.valid:
        GCP_CREDENTIALS.refresh(Request())
        
    return GCP_CREDENTIALS

async def transcribe_audio(audio_base64: str, language_code: str = "en-US", alternative_language_codes: list | None = None) -> str:
    """
    Sends base64 encoded audio to GCP Speech-to-Text via REST.
    Returns the transcription text.
    """
    creds = get_gcp_credentials()
    url = "https://speech.googleapis.com/v1/speech:recognize"
    
    config: dict[str, Any] = {
        "encoding": "LINEAR16",
        "sampleRateHertz": 16000,
        "languageCode": language_code,
        "enableAutomaticPunctuation": True,
        "model": "default"
    }

    if alternative_language_codes:
        config["alternativeLanguageCodes"] = alternative_language_codes

    payload = {
        "audio": {
            "content": audio_base64
        },
        "config": config
    }
    
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Traverse GCP STT response format
        if "results" in data and len(data["results"]) > 0:
            alternatives = data["results"][0].get("alternatives", [])
            if alternatives:
                return alternatives[0].get("transcript", "")
        return ""

async def synthesize_speech(text: str, tone_prompt: str, language_code: str = "en-US") -> str:
    """
    Synthesizes speech using the `gemini-2.5-flash-lite-preview-tts` beta model
    which accepts both `text` and `prompt` (tone).
    Returns base64 encoded audio content.
    """
    creds = get_gcp_credentials()
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    
    # Gemini 2.5 voice has a strict 512 byte limit for text + prompt combined
    # Note: Characters in Arabic/Asian scripts take multiple bytes, so we MUST slice by bytes, not chars.
    clean_prompt = ""
    MAX_PROMPT_BYTES = 50
    MAX_TEXT_BYTES = 400
    
    if tone_prompt and tone_prompt.strip():
        clean_prompt = tone_prompt.strip().encode("utf-8")[:MAX_PROMPT_BYTES].decode("utf-8", "ignore")
        text = text.encode("utf-8")[:MAX_TEXT_BYTES].decode("utf-8", "ignore")
    else:
        text = text.encode("utf-8")[:450].decode("utf-8", "ignore")

    payload: dict[str, Any] = {
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "pitch": 0,
            "speakingRate": 1
        },
        "input": {
            "text": text
        },
        "voice": {
            "languageCode": language_code,
            "modelName": "gemini-2.5-flash-lite-preview-tts",
            "name": "Achernar"
        }
    }
    
    if clean_prompt:
        payload["input"]["prompt"] = clean_prompt
    
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"DEBUG TTS ERROR: {response.text}")
        response.raise_for_status()
        
        data = response.json()
        return data.get("audioContent", "")
