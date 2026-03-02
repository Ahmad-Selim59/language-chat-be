from fastapi import Depends, HTTPException, Header
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
