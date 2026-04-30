"""
Segmentation API endpoints.
Runs VQ-VAE based segmentation on CPU and stores masks in Supabase Storage.
"""

from __future__ import annotations

from datetime import datetime
import io
import logging
import asyncio
from typing import Optional
from uuid import uuid4
from urllib.parse import unquote

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from PIL import Image

from app.config import settings
from app.database.supabase_client import IMAGES_BUCKET, get_supabase_client, get_storage_client
from app.models.segmentation_model import get_segmentation_pipeline

logger = logging.getLogger(__name__)
security = HTTPBearer()
router = APIRouter(prefix="/segmentation", tags=["Segmentation"])


class SegmentationRequest(BaseModel):
    image_file_path: Optional[str] = None
    image_url: Optional[str] = None


class SegmentationResponse(BaseModel):
    original_image_path: str
    original_image_url: str
    segmented_mask_path: str
    segmented_mask_url: str
    reconstructed_image_url: str
    mask_download_name: str
    width: int
    height: int
    tile_size: int
    tiling_used: bool


def _extract_storage_object_path(url_or_path: str) -> str:
    if not url_or_path:
        return ""

    value = url_or_path.strip()
    if value.startswith("http"):
        if f"/{IMAGES_BUCKET}/" in value:
            value = value.split(f"/{IMAGES_BUCKET}/", 1)[1]
        elif f"{IMAGES_BUCKET}/" in value:
            value = value.split(f"{IMAGES_BUCKET}/", 1)[1]

    clean_path = value.split("?", 1)[0].lstrip("/")
    if clean_path.startswith(f"{IMAGES_BUCKET}/"):
        clean_path = clean_path[len(IMAGES_BUCKET) + 1:]

    return unquote(clean_path)


def _create_signed_url(path: str, expires_in_seconds: int = 86400) -> str:
    supabase = get_storage_client()
    signed = supabase.storage.from_(IMAGES_BUCKET).create_signed_url(path=path, expires_in=expires_in_seconds)

    if isinstance(signed, dict):
        url = signed.get("signedURL") or signed.get("signedUrl")
    else:
        url = signed

    if not url:
        return f"{settings.supabase_url.rstrip('/')}/storage/v1/object/public/{IMAGES_BUCKET}/{path}"

    if url.startswith("/"):
        return f"{settings.supabase_url.rstrip('/')}{url}"

    return url


async def _download_from_url(url: str) -> bytes:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=45.0, follow_redirects=True)

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not download image (status {response.status_code})",
            )

        return response.content
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error downloading image from URL: {err}",
        )


async def _download_from_supabase(file_path: str) -> bytes:
    try:
        if file_path.startswith("http"):
            return await _download_from_url(file_path)

        clean_path = _extract_storage_object_path(file_path)
        if not clean_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Supabase file path",
            )

        supabase = get_storage_client()
        last_err = None
        for attempt in range(3):
            try:
                return supabase.storage.from_(IMAGES_BUCKET).download(clean_path)
            except Exception as err:
                last_err = err
                logger.warning(
                    "[SEGMENTATION] Download attempt %s/3 failed for %s: %s",
                    attempt + 1,
                    clean_path,
                    err,
                )
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise last_err if last_err else RuntimeError("Supabase download failed")
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not download image from Supabase: {err}",
        )


def _upload_to_storage(path: str, file_bytes: bytes, content_type: str) -> None:
    supabase = get_storage_client()
    supabase.storage.from_(IMAGES_BUCKET).upload(
        path=path,
        file=file_bytes,
        file_options={"content-type": content_type, "upsert": "true"},
    )


def _resolve_user_id(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    if not credentials:
        return f"anonymous_{uuid4().hex[:16]}"

    try:
        token = credentials.credentials
        supabase = get_supabase_client()
        user_response = supabase.auth.get_user(token)
        if user_response and user_response.user and user_response.user.id:
            return str(user_response.user.id)
    except Exception as err:
        logger.warning("[SEGMENTATION] Failed to resolve user from token: %s", err)

    return f"anonymous_{uuid4().hex[:16]}"


@router.post("/predict", response_model=SegmentationResponse)
async def predict_segmentation(
    req: SegmentationRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    if not req.image_file_path and not req.image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either image_file_path or image_url must be provided",
        )

    user_id = _resolve_user_id(credentials)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    request_id = uuid4().hex

    # Load source image bytes and decide original storage path.
    image_bytes: bytes
    original_path = ""

    if req.image_file_path and not req.image_file_path.startswith("http"):
        try:
            image_bytes = await _download_from_supabase(req.image_file_path)
        except Exception as storage_err:
            logger.warning("[SEGMENTATION] ⚠️ Supabase storage download failed: %s", storage_err)
            if req.image_url:
                logger.info("[SEGMENTATION] Falling back to image_url download")
                image_bytes = await _download_from_url(req.image_url)
            else:
                raise
        original_path = _extract_storage_object_path(req.image_file_path)
    elif req.image_url:
        image_bytes = await _download_from_url(req.image_url)
    elif req.image_file_path and req.image_file_path.startswith("http"):
        image_bytes = await _download_from_url(req.image_file_path)
    else:
        image_bytes = await _download_from_url(req.image_url or "")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image payload: {err}",
        )

    if not original_path:
        original_path = f"{user_id}/segmentations/originals/{timestamp}_{request_id}.png"
        original_png = io.BytesIO()
        image.save(original_png, format="PNG")
        _upload_to_storage(original_path, original_png.getvalue(), "image/png")

    pipeline = get_segmentation_pipeline()
    mask, recon, metadata = pipeline.segment(image)
    mask_png_bytes = pipeline.mask_to_png_bytes(mask)
    reconstructed_base64 = pipeline.image_to_base64(recon)

    mask_path = f"{user_id}/segmentations/masks/{timestamp}_{request_id}_mask.png"
    _upload_to_storage(mask_path, mask_png_bytes, "image/png")

    original_url = _create_signed_url(original_path)
    mask_url = _create_signed_url(mask_path)

    return SegmentationResponse(
        original_image_path=original_path,
        original_image_url=original_url,
        segmented_mask_path=mask_path,
        segmented_mask_url=mask_url,
        reconstructed_image_url=reconstructed_base64,
        mask_download_name=f"segmented_mask_{timestamp}.png",
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        tile_size=int(metadata["tile_size"]),
        tiling_used=bool(metadata["tiling_used"]),
    )
