from fastapi import FastAPI, Body, Query, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.llm_handler import get_bedrock_response, get_chat_name_suggestion, create_translation_for_message
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
)
from fastapi import HTTPException

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
    user_id: str
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
async def send_message(req: ChatRequest, background_tasks: BackgroundTasks) -> dict:
    clean_user_id = clean_id(req.user_id)
    clean_session_id = clean_id(req.session_id)
    clean_user_message = req.user_message.strip()

    # if db_rate_limit_check(clean_user_id):
    #     return "user, used system today already"

    previous_chat_history = await get_chat_history_from_db(
        clean_user_id, clean_session_id
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
        clean_user_id, clean_session_id, clean_user_message, llm_response
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
async def get_chat_history(session_id: str, user_id: str):
    return await get_chat_history_from_db(clean_id(user_id), clean_id(session_id))


@app.get("/sessions")
async def retrieve_all_sessions(user_id: str):
    sessions = await get_sessions_from_db(clean_id(user_id))
    return sessions


@app.delete("/chat")
async def delete_chat(session_id: str = Query(...)):
    await delete_session_from_db(clean_id(session_id))


@app.put("/title")
async def update_session_title(data: TitleUpdate = Body(...)):
    await update_session_title_in_db(clean_id(data.session_id), data.new_title)

@app.get("/translate")
async def translate_message(session_id: str, message_id: int, native_language: str):
    clean_session_id = clean_id(session_id)
    
    existing_translation = await get_message_translation(clean_session_id, message_id, native_language)
    if existing_translation:
        return {"translation": existing_translation}
        
    original_text = await get_message_text(clean_session_id, message_id)
    if not original_text:
        raise HTTPException(status_code=404, detail="Message not found")
    
    translation = await create_translation_for_message(original_text, LLM_MODEL_NAME, native_language)
    
    stored = await save_message_translation(clean_session_id, message_id, native_language, translation)
    
    if not stored:
        # Race condition happened, another request might have generated it concurrently
        # Let's try to fetch it again
        existing_translation = await get_message_translation(clean_session_id, message_id, native_language)
        if existing_translation:
            return {"translation": existing_translation}
    
    return {"translation": translation}