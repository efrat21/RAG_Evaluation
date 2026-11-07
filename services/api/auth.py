"""JWT Authentication and authorization module for the enterprise knowledge assistant.

Provides token generation, verification, user authentication against DEMO_USERS,
FastAPI security dependencies, and the /auth/login route.
"""

from datetime import datetime, timedelta, timezone
import hmac
import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from config import settings
from models import LoginRequest, LoginResponse

logger = logging.getLogger("knowledge_assistant.auth")

router = APIRouter(prefix="/auth", tags=["Authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=True)


def get_demo_users() -> Dict[str, str]:
    """Retrieve demo users dictionary from configuration settings."""
    return settings.parse_demo_users()


def authenticate_user(username: str, password: str) -> bool:
    """Verify username and password against configured demo users.

    Uses constant-time comparison to protect against timing attacks.

    Args:
        username: The user login name.
        password: The plain-text password to verify.

    Returns:
        True if valid, False otherwise.
    """
    users = get_demo_users()
    if username not in users:
        logger.warning("Authentication failed: user '%s' not found", username)
        return False

    expected_password = users[username]
    is_valid = hmac.compare_digest(expected_password.encode("utf-8"), password.encode("utf-8"))
    if not is_valid:
        logger.warning("Authentication failed: invalid password for user '%s'", username)
    return is_valid


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token.

    Args:
        data: Payload claims to encode in token.
        expires_delta: Optional token lifetime. Defaults to JWT_EXPIRE_MINUTES.

    Returns:
        Encoded JWT token string.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": now,
        "nbf": now,
        "iss": settings.OTEL_SERVICE_NAME,
    })

    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Decode and validate a JWT access token.

    Args:
        token: Bearer JWT string.

    Returns:
        Decoded payload dictionary.

    Raises:
        HTTPException: If token is invalid or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        username: Optional[str] = payload.get("sub")
        if username is None:
            logger.warning("Token verification failed: missing 'sub' claim")
            raise credentials_exception
        return payload
    except JWTError as err:
        logger.warning("Token decoding error: %s", err)
        raise credentials_exception from err


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """FastAPI dependency to extract and verify the authenticated username.

    Args:
        token: JWT string extracted from Authorization header.

    Returns:
        Username string of the authenticated user.
    """
    payload = verify_token(token)
    username: str = payload.get("sub", "")
    return username


@router.post("/login", response_model=LoginResponse, summary="User Login")
async def login(credentials: LoginRequest) -> LoginResponse:
    """Authenticate credentials and issue a JWT access token.

    - **username**: Username of the employee or demo account
    - **password**: Corresponding account password
    """
    if not authenticate_user(credentials.username, credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expires_in_seconds = settings.JWT_EXPIRE_MINUTES * 60
    token_expires = timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": credentials.username},
        expires_delta=token_expires,
    )

    logger.info("User '%s' authenticated successfully", credentials.username)
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        username=credentials.username,
        expires_in=expires_in_seconds,
    )
