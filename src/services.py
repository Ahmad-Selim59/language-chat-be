import datetime

from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient

from src.config.env_var import MONGO_URI

MONGO_CLIENT = AsyncIOMotorClient(MONGO_URI)
DATABASE = MONGO_CLIENT["chat_bot"]
CHAT_HISTORY_DB = DATABASE["chat_history"]
MEMORIES_DB = DATABASE["user_memories"]


async def store_chat_in_db(
    user_id: str, session_id: str, user_message: str, llm_response: str
):
    print("Saving to DB...", session_id, user_id, user_message, llm_response)

    current_time = datetime.datetime.utcnow()

    # Try to insert new session but if it already exists append messages instead
    await CHAT_HISTORY_DB.update_one(
        {"_id": session_id},
        {
            "$setOnInsert": {"user_id": user_id},
            "$set": {"updated_at": current_time},
            "$push": {
                "messages": {
                    "$each": [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": llm_response},
                    ]
                }
            },
        },
        upsert=True,
    )


def db_rate_limit_check(user_id: str) -> bool:
    return False


async def get_chat_history_from_db(user_id: str, session_id: str) -> dict:
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


async def update_session_title_in_db(session_id, new_title):
    result = await CHAT_HISTORY_DB.update_one(
        {"_id": session_id}, {"$set": {"title": new_title}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")


async def delete_session_from_db(session_id):
    result = await CHAT_HISTORY_DB.delete_one({"_id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
