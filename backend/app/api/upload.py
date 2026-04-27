"""
Image upload API endpoints using Supabase Storage.
Handles image uploads to Supabase buckets.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.database.supabase_client import get_supabase_client, IMAGES_BUCKET
from app.database.database import get_db
from app.config import settings
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
import uuid
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])
security = HTTPBearer()


class ImageUploadResponse(BaseModel):
    """Response after successful image upload."""
    file_id: str
    file_name: str
    file_path: str
    public_url: str
    uploaded_at: str


@router.post("/image", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Upload an image to Supabase Storage.

    Process:
    1. Verify user (if authenticated)
    2. Generate unique filename
    3. Upload to Supabase Storage bucket
    4. Return public signed URL

    Args:
        file: Image file (JPG, PNG, TIFF, SVS)
        credentials: Optional Bearer token
        db: Database session

    Returns:
        ImageUploadResponse with file URL and metadata

    Raises:
        HTTPException 400: If file type invalid
        HTTPException 500: If upload fails
    """
    supabase = get_supabase_client()
    
    # Extract user ID from token (optional)
    user_id = None
    if credentials:
        try:
            token = credentials.credentials
            logger.info(f"[UPLOAD] Verifying token")
            
            user_response = supabase.auth.get_user(token)
            if user_response and user_response.user:
                user_id = user_response.user.id
                logger.info(f"[UPLOAD] ✅ Authenticated user: {user_id}")
            else:
                logger.warning(f"[UPLOAD] ⚠️ Token invalid, proceeding anonymously")
        except Exception as auth_err:
            logger.warning(f"[UPLOAD] ⚠️ Auth error: {auth_err}, proceeding anonymously")
    
    # For unauthenticated uploads, use a session ID
    if not user_id:
        user_id = f"anonymous_{uuid.uuid4().hex[:16]}"
        logger.info(f"[UPLOAD] 📝 Anonymous session: {user_id}")
    
    # Validate file type
    allowed_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".svs"]
    file_extension = file.filename.lower().split(".")[-1] if "." in file.filename else ""
    
    if f".{file_extension}" not in allowed_extensions:
        logger.error(f"[UPLOAD] ❌ Invalid file type: .{file_extension}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{user_id}/{timestamp}_{file_id}.{file_extension}"
        
        logger.info(f"[UPLOAD] 📤 Starting upload")
        logger.info(f"  Original filename: {file.filename}")
        logger.info(f"  Storage path: {unique_filename}")
        logger.info(f"  Content type: {file.content_type}")
        
        # Read file content
        file_content = await file.read()
        logger.info(f"  File size: {len(file_content)} bytes")
        
        # Upload to Supabase Storage
        logger.info(f"[UPLOAD] 🚀 Uploading to Supabase...")
        upload_response = supabase.storage.from_(IMAGES_BUCKET).upload(
            path=unique_filename,
            file=file_content,
            file_options={"content-type": file.content_type}
        )
        logger.info(f"[UPLOAD] Upload response: {upload_response}")
        
        # Generate signed URL valid for 24 hours
        logger.info(f"[UPLOAD] 🔗 Creating signed URL...")
        signed_url_response = None
        last_sign_error = None
        for attempt in range(3):
            try:
                signed_url_response = supabase.storage.from_(IMAGES_BUCKET).create_signed_url(
                    path=unique_filename,
                    expires_in=86400  # 24 hours
                )
                break
            except Exception as sign_err:
                last_sign_error = sign_err
                logger.warning(f"[UPLOAD] ⚠️ Signed URL attempt {attempt + 1}/3 failed: {sign_err}")
                # Storage propagation can be eventually consistent right after upload.
                await asyncio.sleep(0.6)
        
        # Extract URL from response
        signed_url = None
        if isinstance(signed_url_response, dict):
            signed_url = signed_url_response.get("signedURL") or signed_url_response.get("signedUrl")
        elif isinstance(signed_url_response, str):
            signed_url = signed_url_response
        
        if not signed_url:
            # Fallback: public object URL. Works if bucket/object policy allows read access.
            signed_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/public/{IMAGES_BUCKET}/{unique_filename}"
            logger.warning(
                f"[UPLOAD] ⚠️ Falling back to public URL after signed URL failure: {last_sign_error}"
            )

        # Normalize to absolute URL if needed
        if signed_url.startswith("/"):
            signed_url = f"{settings.supabase_url.rstrip('/')}{signed_url}"
        
        logger.info(f"[UPLOAD] ✅ Upload complete!")
        logger.info(f"  File ID: {file_id}")
        logger.info(f"  Signed URL: {signed_url[:80]}...")
        
        return ImageUploadResponse(
            file_id=file_id,
            file_name=file.filename,
            file_path=unique_filename,
            public_url=signed_url,
            uploaded_at=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPLOAD] ❌ Upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.delete("/image/{file_path:path}")
async def delete_image(
    file_path: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Delete an image from Supabase Storage.

    Args:
        file_path: Path to file in storage
        credentials: Bearer token
        db: Database session

    Returns:
        Success message

    Raises:
        HTTPException 401: If not authenticated
        HTTPException 404: If file not found
    """
    supabase = get_supabase_client()

    try:
        token = credentials.credentials
        logger.info(f"[DELETE] Attempting to delete: {file_path}")
        
        # Verify token
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        # Delete file
        supabase.storage.from_(IMAGES_BUCKET).remove([file_path])
        
        logger.info(f"[DELETE] ✅ File deleted: {file_path}")

        return {"message": "File deleted successfully", "file_path": file_path}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DELETE] Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}"
        )