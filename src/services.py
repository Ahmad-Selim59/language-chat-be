import datetime

from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
from typing import Optional

from src.config.env_var import MONGO_URI, LLM_MODEL_NAME
from src.gcp import transcribe_audio, synthesize_speech
from src.llm_handler import get_bedrock_response, get_response_tone
import json
import os

# Load language map once (Fail hard if not found)
# Flatten it so we can easily map both top-level names AND dialect names
LANG_MAP = {}
with open(os.path.join(os.path.dirname(__file__), "config", "languages.json"), "r") as f:
    langs_data = json.load(f)
    for l in langs_data:
        # Map primary language
        LANG_MAP[l["name"].lower()] = l["code"]
        # Map any dialects too
        if "dialects" in l:
            for d in l["dialects"]:
                LANG_MAP[d["name"].lower()] = d["code"]

MONGO_CLIENT = AsyncIOMotorClient(MONGO_URI)
DATABASE = MONGO_CLIENT["chat_bot"]
CHAT_HISTORY_DB = DATABASE["chat_history"]
MEMORIES_DB = DATABASE["user_memories"]

def clean_id(id_str: str) -> str:
    """Consistently strip quotes and whitespace from IDs."""
    if not id_str:
        return id_str
    return id_str.strip().strip('"').strip("'")


async def store_chat_in_db(
    user_id: str, session_id: str, user_message: str, llm_response: str
):
    print("Saving to DB...", session_id, user_id, user_message, llm_response)

    current_time = datetime.datetime.utcnow()

    # Increment Counter Atomically
    updated_chat = await CHAT_HISTORY_DB.find_one_and_update(
        {"_id": session_id},
        {
            "$setOnInsert": {"user_id": user_id},
            "$set": {"updated_at": current_time},
            "$inc": {"message_counter": 1},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    counter = updated_chat.get("message_counter", 1)
    assistant_message_id = counter

    # Push Message Using That ID
    await CHAT_HISTORY_DB.update_one(
        {"_id": session_id},
        {
            "$push": {
                "messages": {
                    "$each": [
                        {
                            "role": "user",
                            "content": user_message,
                            "created_at": current_time,
                        },
                        {
                            "id": assistant_message_id,
                            "role": "assistant",
                            "content": llm_response,
                            "translation": {}, 
                            "created_at": current_time,
                        },
                    ]
                }
            }
        },
    )


def db_rate_limit_check(user_id: str) -> bool:
    return False


async def get_chat_history_from_db(user_id: str, session_id: str) -> list:
    doc = await CHAT_HISTORY_DB.find_one({"_id": session_id, "user_id": user_id})
    if doc and "messages" in doc:
        return doc["messages"]
    return []


async def get_sessions_from_db(user_id: str) -> dict:
    cursor = CHAT_HISTORY_DB.find({"user_id": user_id}, {"_id": 1, "title": 1}).sort(
        "updated_at", -1
    )
    sessions = []
    async for doc in cursor:
        sessions.append(
            {
                "session_id": str(doc["_id"]),
                "title": doc.get("title", "Untitled Session"),
            }
        )
    return {"sessions": sessions}


async def update_session_title_in_db(session_id, user_id, new_title):
    result = await CHAT_HISTORY_DB.update_one(
        {"_id": session_id, "user_id": user_id}, {"$set": {"title": new_title}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")


async def delete_session_from_db(session_id, user_id):
    result = await CHAT_HISTORY_DB.delete_one({"_id": session_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

async def set_chat_name_in_db(session_id, chat_name):
    session_id = clean_id(session_id)
    # Only update if the title is missing or is the default "Untitled Session"
    # This prevents the background LLM task from overwriting a manual user update
    result = await CHAT_HISTORY_DB.update_one(
        {
            "_id": session_id, 
            "$or": [
                {"title": {"$exists": False}},
                {"title": "Untitled Session"}
            ]
        }, 
        {"$set": {"title": chat_name}}
    )
    
    if result.matched_count > 0:
        print(f"\n\nChat name suggested and set: {session_id} -> {chat_name}")
    else:
        print(f"\n\nChat name suggestion skipped (title already exists): {session_id}")

async def get_message_text(session_id: str, user_id: str, message_id: int) -> Optional[str]:
    doc = await CHAT_HISTORY_DB.find_one(
        {
            "_id": session_id,
            "user_id": user_id,
            "messages": {"$elemMatch": {"id": message_id}}
        },
        {"messages.$": 1}
    )
    if doc and doc.get("messages"):
        content = doc["messages"][0].get("content")
        return str(content) if content else None
    return None

async def get_message_translation(session_id: str, user_id: str, message_id: int, target_language: str) -> Optional[str]:
    doc = await CHAT_HISTORY_DB.find_one(
        {
            "_id": session_id,
            "user_id": user_id,
            "messages": {"$elemMatch": {"id": message_id}}
        },
        {"messages.$": 1}
    )
    if not doc or not doc.get("messages"):
        return None
        
    message = doc["messages"][0]
    
    translation_dict = message.get("translation", {})
    if isinstance(translation_dict, dict) and target_language in translation_dict:
        return str(translation_dict[target_language])
        
    return None

async def save_message_translation(session_id: str, user_id: str, message_id: int, target_language: str, translation_text: str) -> bool:
    # Conditionally update to prevent race conditions
    # This will only match if the translation.target_language field DOES NOT exist
    result = await CHAT_HISTORY_DB.update_one(
        {
            "_id": session_id,
            "user_id": user_id,
            "messages.id": message_id,
            f"messages.translation.{target_language}": {"$exists": False}
        },
        {
            "$set": {
                f"messages.$.translation.{target_language}": translation_text
            }
        }
    )
    return result.modified_count > 0

def _map_to_bcp47(lang_name: str) -> str:
    return LANG_MAP.get(str(lang_name).lower(), "en-US")

async def process_oral_chat_message(audio_base64: str, history: list, settings: dict, websocket=None) -> dict:
    """Processes a single audio message: STT -> LLM -> Tone -> TTS. Streams intermediate results to websocket."""
    target_lang_name = settings.get("targetLanguage", "English")
    # Dialect name takes precedence if provided, otherwise use the target language name.
    effective_target_lang = settings.get("dialect") or target_lang_name
    effective_native_lang = settings.get("nativeLanguage", "English")

    native_bcp47 = _map_to_bcp47(effective_native_lang)
    target_bcp47 = _map_to_bcp47(effective_target_lang)
    
    # Restored multi-language detection.
    # Google will prioritize target_bcp47 but allow native fallback if the sound matches better.
    alt_langs = [native_bcp47] if native_bcp47 != target_bcp47 else None

    # We force the recognizer strictly on the target language to prevent it from matching English words blindly.
    user_text = await transcribe_audio(
        audio_base64, 
        language_code=target_bcp47,
        alternative_language_codes=alt_langs
    )
    
    if not user_text:
        if websocket:
            await websocket.send_json({"error": "Failed to transcribe audio"})
        return {}
        
    if websocket:
        await websocket.send_json({
            "type": "user_transcription",
            "content": user_text
        })
        
    llm_response = await get_bedrock_response(
        prompt=user_text,
        model=LLM_MODEL_NAME,
        previous_chat_history=history,
        settings=settings
    )
    
    tone_prompt = await get_response_tone(llm_response, LLM_MODEL_NAME)
    
    # AI character should reply in the target language (e.g., Malay)
    ai_audio_base64 = await synthesize_speech(llm_response, tone_prompt, language_code=target_bcp47)
    
    result = {
        "type": "assistant_response",
        "content": llm_response,
        "audio": ai_audio_base64
    }
    
    if websocket:
        await websocket.send_json(result)
        
    return result