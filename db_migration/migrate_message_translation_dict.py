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
    
    print("Running migration to convert translation strings to dicts...")
    
    cursor = chat_history_db.find({})
    modified_count = 0
    matched_count = 0
    
    async for doc in cursor:
        matched_count += 1
        messages = doc.get("messages", [])
        updated_messages = []
        changed = False
        
        for msg in messages:
            new_msg = dict(msg) 
            
            if new_msg.get("role") == "assistant":
                current_translation = new_msg.get("translation")
                if isinstance(current_translation, str):
                    changed = True
                    # Convert string to dict. Since they were previously empty strings, we just set to empty dict.
                    new_msg["translation"] = {}
                    
            updated_messages.append(new_msg)
            
        if changed:
            res = await chat_history_db.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "messages": updated_messages
                    }
                }
            )
            if res.modified_count > 0:
                modified_count += 1
            
    print(f"Migration complete: Matched {matched_count} documents and modified {modified_count} documents.")

if __name__ == "__main__":
    asyncio.run(main())
