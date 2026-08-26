import logging
import os
from dataclasses import dataclass

import jwt
from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

# JWT Secret Key
JWT_SECRET = os.getenv("JWT_SECRET")


@dataclass
class AuthContext:
    """Resolved identity for a request, passed to route handlers via Depends(token_required)."""
    user_id: str
    token: str


def _decode_user_id(token: str) -> str:
    data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"verify_sub": False})
    return data["user_id"]


def _strip_bearer(auth_header: str) -> str:
    return auth_header.split()[1] if "Bearer" in auth_header else auth_header


async def token_required(authorization: str = Header(None)) -> AuthContext:
    """FastAPI dependency: validates the Authorization header and resolves the user.

    Use as `auth: AuthContext = Depends(token_required)` on any route that needs it.
    """
    if not authorization:
        raise HTTPException(status_code=403, detail="Token is missing!")

    try:
        token = _strip_bearer(authorization)
        current_user_id = _decode_user_id(token)
    except Exception as e:
        logging.error(f"Error decoding token: {e}")
        raise HTTPException(status_code=403, detail="Token is invalid!")

    return AuthContext(user_id=current_user_id, token=token)


def decode_socket_token(auth_header: str) -> AuthContext:
    """Validates a token supplied at Socket.IO connect time.

    Framework-agnostic on purpose: raises ValueError on failure and lets the
    caller (the socket connect handler) decide how to reject the connection,
    instead of reaching into a specific socket library itself.
    """
    if not auth_header:
        raise ValueError("No Authorization header found")

    token = _strip_bearer(auth_header)
    current_user_id = _decode_user_id(token)
    return AuthContext(user_id=current_user_id, token=token)
