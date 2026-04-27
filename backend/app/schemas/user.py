"""
Pydantic schemas for request/response validation.
These define the structure of data sent to and from API endpoints.
"""

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    """
    Schema for user signup request.
    
    Used when user creates a new account.
    """
    full_name: str = Field(..., min_length=2, max_length=255, example="John Doe")
    email: EmailStr = Field(..., example="john.doe@example.com")
    password: str = Field(..., min_length=8, max_length=100, example="SecurePassword123!")


class UserLogin(BaseModel):
    """
    Schema for user login request.
    
    Used when user logs in.
    """
    email: EmailStr = Field(..., example="john.doe@example.com")
    password: str = Field(..., example="SecurePassword123!")


class UserResponse(BaseModel):
    """
    Schema for user data in responses.

    NEVER includes password or password_hash!
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    full_name: Optional[str] = None
    email: str
    created_at: datetime
    last_login: Optional[datetime] = None


class TokenResponse(BaseModel):
    """
    Schema for authentication token response.
    
    Returned after successful signup or login.
    """
    access_token: str
    token_type: str = "bearer"
    refresh_token: str
    expires_in: int
    user: UserResponse


class MessageResponse(BaseModel):
    """
    Generic message response.
    
    Used for simple success/error messages.
    """
    message: str