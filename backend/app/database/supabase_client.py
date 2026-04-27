"""
Supabase client initialization and helpers.
Handles authentication and storage operations.
"""

from supabase import create_client, Client
from app.config import settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase: Client = create_client(settings.supabase_url, settings.supabase_service_key)


def get_supabase_client() -> Client:
    """
    Get Supabase client instance.
    
    Returns:
        Supabase client for database and storage operations
    """
    return supabase


# Storage bucket name for images
IMAGES_BUCKET = "cytosight-images"


def initialize_storage():
    """
    Initialize storage bucket if it doesn't exist.
    Call this on application startup.
    """
    try:
        # Check if bucket exists
        buckets = supabase.storage.list_buckets()
        bucket_names = [bucket.name for bucket in buckets]
        
        if IMAGES_BUCKET not in bucket_names:
            # Create bucket as public
            supabase.storage.create_bucket(
                IMAGES_BUCKET,
                options={"public": True}
            )
            logger.info(f"Created public storage bucket: {IMAGES_BUCKET}")
        else:
            # Ensure the bucket is public if it already exists
            supabase.storage.update_bucket(
                IMAGES_BUCKET,
                options={"public": True}
            )
            logger.info(f"Updated storage bucket to public: {IMAGES_BUCKET}")
    except Exception as e:
        logger.error(f"Error initializing storage: {e}")