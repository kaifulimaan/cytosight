"""
Authentication API endpoints using Supabase.
Handles user signup, login, and session management.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.database.supabase_client import get_supabase_client
from app.database.database import get_db
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)

# Import schemas if they exist
try:
    from app.schemas.user import UserCreate, UserLogin, UserResponse
except ImportError:
    # Fallback models if schemas don't exist
    class UserCreate(BaseModel):
        full_name: str
        email: str
        password: str

    class UserLogin(BaseModel):
        email: str
        password: str

    class UserResponse(BaseModel):
        id: str
        email: str
        full_name: Optional[str] = None
        created_at: Optional[str] = None
        last_sign_in_at: Optional[str] = None

        class Config:
            from_attributes = True

from typing import Optional

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


class TokenResponse(BaseModel):
    """Authentication response with token."""
    access_token: str
    token_type: str = "bearer"
    refresh_token: str
    user: dict  # Simplified to dict for flexibility
    expires_in: int = 3600


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user with Supabase Auth.
    
    Process:
    1. Create user in Supabase Auth (handles password hashing automatically)
    2. Return access token and user info
    
    Args:
        user_data: User signup data (full_name, email, password)
        db: Database session
        
    Returns:
        TokenResponse with access token and user info
        
    Raises:
        HTTPException 400: If email already registered
        HTTPException 500: If signup fails
    """
    supabase = get_supabase_client()
    
    try:
        # Normalize email to lowercase
        email = user_data.email.strip().lower()
        password = user_data.password.strip()
        full_name = user_data.full_name.strip()
        
        logger.info(f"[SIGNUP] Attempting to register: {email}")
        
        # Sign up user with Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name
                }
            }
        })

        if not auth_response.user:
            logger.error(f"[SIGNUP] User creation failed for {email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User registration failed. Email may already be registered."
            )

        user = auth_response.user
        session = auth_response.session

        # Check if email confirmation is required
        if not session:
            logger.warning(f"[SIGNUP] Email confirmation required for {email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signup successful! Please check your email to confirm your account."
            )

        # Prepare user response
        user_response = {
            "id": user.id,
            "email": user.email,
            "full_name": user.user_metadata.get("full_name") if user.user_metadata else full_name,
            "created_at": str(user.created_at) if user.created_at else None,
            "last_sign_in_at": str(user.last_sign_in_at) if user.last_sign_in_at else None
        }

        logger.info(f"[SIGNUP]  User registered: {email}")

        return TokenResponse(
            access_token=session.access_token,
            token_type="bearer",
            refresh_token=session.refresh_token,
            user=user_response,
            expires_in=session.expires_in
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e).lower()
        logger.error(f"[SIGNUP] Error: {e}")
        
        # Check for duplicate email error
        if "already registered" in error_str or "duplicate" in error_str or "user already exists" in error_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signup failed: {str(e)}"
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return session token.
    
    Process:
    1. Authenticate with Supabase Auth
    2. Supabase verifies password automatically
    3. Return access token and user info
    
    Args:
        credentials: Login credentials (email, password)
        db: Database session
        
    Returns:
        TokenResponse with access token and user info
        
    Raises:
        HTTPException 401: If credentials are invalid
    """
    supabase = get_supabase_client()
    
    try:
        # Normalize email to lowercase
        email = credentials.email.strip().lower()
        password = credentials.password.strip()
        
        logger.info(f"[LOGIN] Attempting login for: {email}")
        
        # Sign in with Supabase Auth
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if not auth_response.user or not auth_response.session:
            logger.error(f"[LOGIN] Failed login attempt for: {email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        user = auth_response.user
        session = auth_response.session
        
        # Prepare user response
        user_response = {
            "id": user.id,
            "email": user.email,
            "full_name": user.user_metadata.get("full_name") if user.user_metadata else None,
            "created_at": str(user.created_at) if user.created_at else None,
            "last_sign_in_at": str(user.last_sign_in_at) if user.last_sign_in_at else None
        }
        
        logger.info(f"[LOGIN]  Successful login for: {email}")
        
        return TokenResponse(
            access_token=session.access_token,
            token_type="bearer",
            refresh_token=session.refresh_token,
            user=user_response,
            expires_in=session.expires_in
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LOGIN] Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Log out current user.
    
    Note: JWT tokens are stateless, so logout is client-side by removing token.
    This endpoint is for server-side invalidation if needed.
    
    Returns:
        Success message
    """
    supabase = get_supabase_client()
    
    try:
        token = credentials.credentials
        logger.info("[LOGOUT] User logging out")
        
        supabase.auth.sign_out(token)
        
        return {"message": "Logged out successfully"}
    except Exception as e:
        logger.warning(f"[LOGOUT] Error (continuing): {e}")
        # Return success even if sign_out fails
        return {"message": "Logged out successfully"}


@router.get("/me", response_model=dict)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get current user information from token.
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        User info
        
    Raises:
        HTTPException 401: If token is invalid
    """
    supabase = get_supabase_client()
    
    try:
        token = credentials.credentials
        logger.info("[GET_USER] Fetching current user")
        
        # Get user from token
        user_response = supabase.auth.get_user(token)
        
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        user = user_response.user
        
        return {
            "id": user.id,
            "email": user.email,
            "full_name": user.user_metadata.get("full_name") if user.user_metadata else None,
            "created_at": str(user.created_at) if user.created_at else None,
            "last_sign_in_at": str(user.last_sign_in_at) if user.last_sign_in_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GET_USER] Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )