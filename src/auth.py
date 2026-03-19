from fastapi import Depends, HTTPException, Header, WebSocket, WebSocketDisconnect
from jwt import PyJWKClient
import jwt

from src.config.env_var import SUPABASE_PROJECT_URI

SUPABASE_JWKS_URL = f"{SUPABASE_PROJECT_URI}/auth/v1/.well-known/jwks.json"

# Initialized once and reuses cached keys
_jwks_client = PyJWKClient(SUPABASE_JWKS_URL)


def _decode_token(token: str) -> dict:
    """Verify and decode a Supabase JWT using the public JWKS endpoint."""
    try:
        # Get the signing key from the JWKS endpoint (supports RS256, ES256, etc.)
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        
        # Decode using the algorithm specified in the token header
        # We explicitly allow ES256 here since your project is using it
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256"],
            audience="authenticated",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        # Removing the token from the error message for security in production
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication error: {e}")


def get_current_user_id(authorization: str | None = Header(None, alias="Authorization")) -> str:
    """
    FastAPI dependency that extracts and verifies the Bearer token from the
    Authorization header, then returns the Supabase user UUID (sub claim).
    """
    if not authorization:
         raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must start with 'Bearer '")

    token = authorization.removeprefix("Bearer ").strip()
    payload = _decode_token(token)

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token is missing 'sub' claim")

    return user_id

async def authenticate_oral_chat(websocket: WebSocket) -> str | None:
    """Handles the initial authentication message over the WebSocket."""
    try:
        auth_msg = await websocket.receive_json()
        if auth_msg.get("type") != "auth" or not auth_msg.get("token"):
            await websocket.close(code=1008, reason="Missing authentication token")
            return None
            
        token = auth_msg.get("token")
        payload = _decode_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            await websocket.close(code=1008, reason="Invalid token")
            return None
        return user_id
    except WebSocketDisconnect:
        print("WebSocket disconnected during auth.")
        return None
    except HTTPException as e:
        print(f"WebSocket auth failed (HTTPException): {e.detail}")
        try:
            await websocket.close(code=1008, reason=str(e.detail))
        except RuntimeError:
            pass
        return None
    except Exception as e:
        print(f"WebSocket auth failed: {e}")
        try:
            await websocket.close(code=1008, reason="Authentication failed")
        except RuntimeError:
            pass
        return None