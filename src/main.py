from fastapi import FastAPI, Body, Query, BackgroundTasks, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.auth import get_current_user_id, authenticate_oral_chat
from src.llm_handler import get_bedrock_response, get_chat_name_suggestion, create_translation_for_message, get_response_tone
from src.gcp import transcribe_audio, synthesize_speech
from src.config.env_var import LLM_MODEL_NAME
from src.services import (
    store_chat_in_db,
    db_rate_limit_check,
    get_chat_history_from_db,
    get_sessions_from_db,
    update_session_title_in_db,
    delete_session_from_db,
    set_chat_name_in_db,
    clean_id,
    get_message_text,
    get_message_translation,
    save_message_translation,
    process_oral_chat_message,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://language-chat-buddy-fe.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Settings(BaseModel):
    targetLanguage: str
    nativeLanguage: str
    scriptPreference: str
    formality: str
    gender: str
    dialect: str


class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    settings: Settings


class TitleUpdate(BaseModel):
    session_id: str
    new_title: str


@app.get("/")
async def read_root():
    pass


async def suggest_and_set_chat_name(session_id: str, user_message: str):
    """Background task to suggest and set a chat name."""
    chat_name_suggestion = await get_chat_name_suggestion(
        user_message,
        LLM_MODEL_NAME,
    )
    await set_chat_name_in_db(session_id, chat_name_suggestion)


@app.post("/chat")
async def send_message(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    clean_session_id = clean_id(req.session_id)
    clean_user_message = req.user_message.strip()

    # if db_rate_limit_check(clean_user_id):
    #     return "user, used system today already"

    previous_chat_history = await get_chat_history_from_db(
        user_id, clean_session_id
    )

    settings_dict = {
        "targetLanguage": req.settings.targetLanguage,
        "nativeLanguage": req.settings.nativeLanguage,
        "scriptPreference": req.settings.scriptPreference,
        "formality": req.settings.formality,
        "gender": req.settings.gender,
        "dialect": req.settings.dialect,
    }

    llm_response = await get_bedrock_response(
            clean_user_message,
            LLM_MODEL_NAME,
            previous_chat_history,
            settings_dict,
        )

    await store_chat_in_db(
        user_id, clean_session_id, clean_user_message, llm_response
    )

    # Suggest a name only for the first message in the background
    if not previous_chat_history:
        background_tasks.add_task(
            suggest_and_set_chat_name, 
            clean_session_id, 
            clean_user_message
        )

    return {"llm_response": llm_response}


@app.get("/chat")
async def get_chat_history(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
):
    return await get_chat_history_from_db(user_id, clean_id(session_id))


@app.get("/sessions")
async def retrieve_all_sessions(user_id: str = Depends(get_current_user_id)):
    sessions = await get_sessions_from_db(user_id)
    return sessions


@app.delete("/chat")
async def delete_chat(
    session_id: str = Query(...),
    user_id: str = Depends(get_current_user_id),
):
    clean_session_id = clean_id(session_id)
    if not clean_session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")
        
    await delete_session_from_db(clean_session_id, user_id)


@app.put("/title")
async def update_session_title(
    data: TitleUpdate = Body(...),
    user_id: str = Depends(get_current_user_id),
):
    await update_session_title_in_db(clean_id(data.session_id), user_id, data.new_title)


@app.get("/translate")
async def translate_message(
    session_id: str,
    message_id: int,
    native_language: str,
    user_id: str = Depends(get_current_user_id),
):
    clean_session_id = clean_id(session_id)
    if not clean_session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    existing_translation = await get_message_translation(clean_session_id, user_id, message_id, native_language)
    if existing_translation:
        return {"translation": existing_translation}
        
    original_text = await get_message_text(clean_session_id, user_id, message_id)
    if not original_text:
        raise HTTPException(status_code=404, detail="Message not found")
    
    translation = await create_translation_for_message(original_text, LLM_MODEL_NAME, native_language)
    
    stored = await save_message_translation(clean_session_id, user_id, message_id, native_language, translation)
    
    if not stored:
        # Race condition: another request already stored a translation, fetch and return it
        existing_translation = await get_message_translation(clean_session_id, user_id, message_id, native_language)
        if existing_translation:
            return {"translation": existing_translation}
    
    return {"translation": translation}


@app.websocket("/ws/oral-chat/{session_id}")
async def oral_chat_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    user_id = await authenticate_oral_chat(websocket)
    if not user_id:
        return

    # 2. Main oral chat loop
    clean_sess_id = clean_id(session_id)
    try:
        while True:
            data = await websocket.receive_json()
            audio_base64 = data.get("audio_base64")
            history = data.get("history", [])
            settings = data.get("settings", {})
            
            if not audio_base64:
                await websocket.send_json({"error": "No audio_base64 provided"})
                continue
            
            await process_oral_chat_message(audio_base64, history, settings, websocket=websocket)
            
    except WebSocketDisconnect:
        print(f"Client disconnected from oral-chat API (session: {clean_sess_id})")
    except Exception as e:
        print(f"Unexpected error in oral-chat websocket: {e}")
        try:
            await websocket.close(code=1011)
        except:
            pass