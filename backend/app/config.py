"""
Configuration settings for Supabase integration.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Supabase
    supabase_url: str
    supabase_key: str
    supabase_service_key: str
    
    # Database (fallback for direct queries if needed)
    database_url: str
    
    # JWT (for custom tokens if needed)
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    
    # HuggingFace (for model downloads)
    hf_token: str = ""  # Optional - for downloading models from HuggingFace
    
    # OpenAI (for explainability)
    openai_api_key: str = ""
    
    # Environment
    environment: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra environment variables without error


@lru_cache()
def get_settings():
    """Get cached settings instance."""
    return Settings()


settings = get_settings()