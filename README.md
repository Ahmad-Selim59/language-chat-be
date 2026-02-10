Steps to Run Your Server in a screen Session

1. Create a New Screen Session
```
screen -S therapist-api
```

This opens a new terminal session named therapist-api.

2. Run Your Server Inside Screen
In the new screen:
```
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
```
Use --host 0.0.0.0 to make it accessible externally (not just 127.0.0.1).

3. Detach from Screen (Keep it Running)
Press:
```
Ctrl + A, then D
```

This detaches the session and leaves your server running in the background.

4. List and Reattach to a Screen Session
List running screens:
```
screen -ls
```

Reattach to your session:
```
screen -r therapist-api
```

env vars:

    MONGO_URI 
    AWS_ACCESS_KEY_ID 
    AWS_SECRET_ACCESS_KEY
    AWS_REGION_NAME 
    LLM_MODEL_NAME 

Poetry being used
poetry run uvicorn src.main:app --reload