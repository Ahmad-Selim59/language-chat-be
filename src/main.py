from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query

from src.llm_handler import get_bedrock_response
from src.config.env_var import LLM_MODEL_NAME
from src.services import (
    store_chat_in_db,
    db_rate_limit_check,
    get_chat_history_from_db,
    get_sessions_from_db,
    update_session_title_in_db,
    delete_session_from_db,
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


@app.post("/chat")
async def send_message(req: ChatRequest) -> dict:
    clean_user_id = req.user_id.strip('"')
    clean_session_id = req.session_id.strip('"')
    clean_user_message = req.user_message.strip('"')

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
    }

    if previous_chat_history:
        llm_response = get_bedrock_response(
            clean_user_message,
            LLM_MODEL_NAME,
            previous_chat_history,
            settings_dict,
        )
    else:
        llm_response = get_bedrock_response(
            clean_user_message,
            LLM_MODEL_NAME,
            previous_chat_history,
            settings_dict,
        )

    await store_chat_in_db(
        clean_user_id, clean_session_id, clean_user_message, llm_response
    )

    return {"llm_response": llm_response}


@app.get("/chat")
async def get_chat_history(session_id: str, user_id: str):
    return await get_chat_history_from_db(user_id, session_id)


@app.get("/sessions")
async def retrieve_all_sessions(user_id: str):
    sessions = await get_sessions_from_db(user_id)
    return sessions


@app.delete("/chat")
async def delete_chat(session_id: str = Query(...)):
    await delete_session_from_db(session_id)


@app.put("/title")
async def update_session_title(data: TitleUpdate = Body(...)):
    await update_session_title_in_db(data.session_id, data.new_title)
