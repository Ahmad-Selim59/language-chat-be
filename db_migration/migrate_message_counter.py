import asyncio
import os
import sys

# Add project root to python path to import src.config.env_var
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from motor.motor_asyncio import AsyncIOMotorClient
from src.config.env_var import MONGO_URI

async def main():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client["chat_bot"]
    chat_history_db = db["chat_history"]
    
    print("Running migration for message_counter and message structures...")
    
    cursor = chat_history_db.find({})
    modified_count = 0
    matched_count = 0
    
    async for doc in cursor:
        matched_count += 1
        messages = doc.get("messages", [])
        assistant_counter = 0
        updated_messages = []
        
        for msg in messages:
            # Create a copy to avoid mutating the original dict unexpectedly
            new_msg = dict(msg) 
            
            if new_msg.get("role") == "assistant":
                assistant_counter += 1
                if "id" not in new_msg:
                    new_msg["id"] = assistant_counter
                if "translation" not in new_msg:
                    new_msg["translation"] = ""
                    
            updated_messages.append(new_msg)
            
        res = await chat_history_db.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "message_counter": assistant_counter,
                    "messages": updated_messages
                }
            }
        )
        if res.modified_count > 0:
            modified_count += 1
            
    print(f"Migration complete: Matched {matched_count} documents and modified {modified_count} documents.")

if __name__ == "__main__":
    asyncio.run(main())
