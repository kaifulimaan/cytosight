"""
Diagnosis API endpoints.
Handles inference requests and returns disease classification results.
Includes history management for past diagnoses.
"""

from fastapi import APIRouter, HTTPException, status, File, UploadFile, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import logging
import torch
import io
import httpx
import json
import asyncio
from urllib.parse import unquote
from datetime import datetime
from uuid import uuid4
from sqlalchemy.orm import Session

from app.models.disease_model import get_model
from app.config import settings
from app.utils.preprocessing import ImagePreprocessor, load_image_from_bytes, extract_tiles_from_image
from app.utils.label_mapping import get_label_mapper
from app.database.models import Diagnosis, User
from app.database.database import get_db
from app.database.supabase_client import get_supabase_client, IMAGES_BUCKET

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])

# Match Kaggle inference settings exactly
TILE_SIZE = 256
MAX_TILES = 500

# Kaggle standardize_transform: Resize entire image to 256x256
import torchvision.transforms as tv_transforms
_standardize_transform = tv_transforms.Resize((TILE_SIZE, TILE_SIZE))


def prepare_tiles(image_bytes: bytes, preprocessor: ImagePreprocessor) -> List[torch.Tensor]:
    """
    Extract and preprocess tiles from image bytes.
    Matches Kaggle SimpleSlideDataset.__getitem__ EXACTLY:

    For regular images (png/jpg/jpeg/tiff):
      - Resize the ENTIRE image to 256x256 as ONE single tile
      - This matches: tiles = [standardize_transform(Image.open(path).convert('RGB'))]

    For WSI formats (svs/ndpi):
      - Extract grid of 256x256 tiles from the slide

    Then apply test_transform (Resize(224), ToTensor, Normalize) to each tile.

    Args:
        image_bytes: Raw image bytes
        preprocessor: ImagePreprocessor instance

    Returns:
        List containing one tensor of shape (num_tiles, 3, 224, 224)
    """
    image = load_image_from_bytes(image_bytes)

    # Match Kaggle SimpleSlideDataset exactly:
    # For regular images: resize whole image to 256x256 as ONE tile
    # The Kaggle code does: tiles = [standardize_transform(Image.open(path).convert('RGB'))]
    # It does NOT split regular images into a grid of crops!
    raw_tiles = [_standardize_transform(image)]

    logger.info(f"[TILES] Prepared {len(raw_tiles)} tile(s) from image ({image.width}x{image.height})")

    # Apply same transform as Kaggle test_transform:
    #   transforms.Resize(224), transforms.ToTensor(),
    #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tile_tensors = torch.stack([
        preprocessor.transform(tile) for tile in raw_tiles
    ])  # shape: (num_tiles, 3, 224, 224)

    logger.info(f"[TILES] Final tensor shape: {tile_tensors.shape}")

    # Wrap in list as model.forward() expects list of tile batches
    return [tile_tensors]


class DiagnosisRequest(BaseModel):
    """Request body for diagnosis prediction."""
    image_file_path: Optional[str] = None  # Path in Supabase storage
    image_url: Optional[str] = None  # Full URL to image


class DiagnosisPrediction(BaseModel):
    """Single prediction result."""
    index: int
    name: str
    confidence: float


class DiseaseInfo(BaseModel):
    """Disease classification result."""
    index: int
    key: str
    name: str
    description: str
    confidence: float


class SeverityInfo(BaseModel):
    """Severity classification result."""
    index: int
    name: str
    description: str
    level: int
    confidence: float


class StageInfo(BaseModel):
    """Stage classification result (if applicable)."""
    index: int
    name: str
    description: str


class DiagnosisResponse(BaseModel):
    """Complete diagnosis response."""
    disease: DiseaseInfo
    severity: SeverityInfo
    stage: Optional[StageInfo] = None
    region: str = "microscopic_tissue"
    status: str  # "normal" or "abnormal"
    diagnosis: str  # Human-readable summary
    confidence_score: float
    model_version: str = "phase3_mil"
    diagnosis_id: Optional[str] = None


class DiagnosisHistoryItem(BaseModel):
    """Single diagnosis from history."""
    diagnosis_id: str
    region: str
    disease_name: str
    severity: str
    stage: Optional[str] = None
    confidence_disease: float
    confidence_severity: float
    confidence_stage: Optional[float] = None
    original_image_url: str
    created_at: str
    status: str


class DiagnosisHistoryResponse(BaseModel):
    """User's complete diagnosis history."""
    total_diagnoses: int
    diagnoses: List[DiagnosisHistoryItem]


def _build_history_item_from_record(record: Diagnosis) -> DiagnosisHistoryItem:
    diagnosis_status = "abnormal" if str(record.severity).strip().lower() == "abnormal" else "normal"
    region = record.disease_name or "unknown"
    return DiagnosisHistoryItem(
        diagnosis_id=str(record.id),
        region=region,
        disease_name=record.disease_name,
        severity=record.severity,
        stage=record.stage,
        confidence_disease=record.confidence_disease or 0.0,
        confidence_severity=record.confidence_severity or 0.0,
        confidence_stage=record.confidence_stage,
        original_image_url=_to_history_image_url(record.original_image_url or ""),
        created_at=record.created_at.isoformat() if record.created_at else "",
        status=diagnosis_status,
    )


def _build_history_item_from_dict(record: dict) -> DiagnosisHistoryItem:
    severity_value = record.get("severity") or "Unknown"
    diagnosis_status = "abnormal" if str(severity_value).strip().lower() == "abnormal" else "normal"
    region = record.get("disease_name") or "unknown"
    return DiagnosisHistoryItem(
        diagnosis_id=str(record.get("id", "")),
        region=region,
        disease_name=record.get("disease_name") or "Unknown",
        severity=severity_value,
        stage=record.get("stage"),
        confidence_disease=float(record.get("confidence_disease") or 0.0),
        confidence_severity=float(record.get("confidence_severity") or 0.0),
        confidence_stage=(
            float(record.get("confidence_stage"))
            if record.get("confidence_stage") is not None
            else None
        ),
        original_image_url=_to_history_image_url(record.get("original_image_url") or ""),
        created_at=str(record.get("created_at") or ""),
        status=diagnosis_status,
    )


def _save_diagnosis_via_supabase(user_id: str, payload: dict) -> Optional[str]:
    """Fallback persistence path when SQLAlchemy DB is unavailable."""
    supabase = get_supabase_client()
    response = supabase.table("diagnoses").insert(payload).execute()
    rows = getattr(response, "data", None) or []
    if rows and rows[0].get("id"):
        return str(rows[0]["id"])
    return None


def _extract_storage_object_path(url_or_path: str) -> str:
    """Extract bucket-relative object path from full URL or raw path."""
    if not url_or_path:
        return ""

    value = str(url_or_path).strip()
    if not value:
        return ""

    if value.startswith("http"):
        if f"/{IMAGES_BUCKET}/" in value:
            value = value.split(f"/{IMAGES_BUCKET}/", 1)[1]
        elif f"{IMAGES_BUCKET}/" in value:
            value = value.split(f"{IMAGES_BUCKET}/", 1)[1]

    clean_path = value.split("?", 1)[0].lstrip("/")
    if clean_path.startswith(f"{IMAGES_BUCKET}/"):
        clean_path = clean_path[len(IMAGES_BUCKET) + 1:]

    return unquote(clean_path)


def _to_history_image_url(url_or_path: str) -> str:
    """Return a frontend-displayable URL for history images."""
    if not url_or_path:
        return ""

    # If it's NOT a Supabase URL, return as is
    if url_or_path.startswith("http") and IMAGES_BUCKET not in url_or_path:
        return url_or_path

    clean_path = _extract_storage_object_path(url_or_path)
    if not clean_path:
        return url_or_path

    try:
        supabase = get_supabase_client()
        signed_url_response = supabase.storage.from_(IMAGES_BUCKET).create_signed_url(
            path=clean_path,
            expires_in=86400,  # 24 hours
        )

        signed_url = None
        if isinstance(signed_url_response, dict):
            signed_url = signed_url_response.get("signedURL") or signed_url_response.get("signedUrl")
        elif isinstance(signed_url_response, str):
            signed_url = signed_url_response

        if signed_url and signed_url.startswith("/"):
            signed_url = f"{settings.supabase_url.rstrip('/')}{signed_url}"

        return signed_url or url_or_path
    except Exception as err:
        logger.warning(f"[HISTORY] ⚠️ Could not sign image path {clean_path}: {err}")
        return url_or_path


def _save_diagnosis_to_storage_history(user_id: str, payload: dict) -> Optional[str]:
    """Last-resort persistence path using Supabase Storage JSON files."""
    supabase = get_supabase_client()
    diagnosis_id = str(payload.get("id") or uuid4())
    created_at = payload.get("created_at") or datetime.utcnow().isoformat()

    record = {
        "id": diagnosis_id,
        "user_id": str(user_id),
        "disease_name": payload.get("disease_name"),
        "severity": payload.get("severity"),
        "stage": payload.get("stage"),
        "confidence_disease": payload.get("confidence_disease"),
        "confidence_severity": payload.get("confidence_severity"),
        "confidence_stage": payload.get("confidence_stage"),
        "original_image_url": payload.get("original_image_url") or "",
        "created_at": created_at,
    }

    history_path = f"history/{user_id}/{diagnosis_id}.json"
    supabase.storage.from_(IMAGES_BUCKET).upload(
        path=history_path,
        file=json.dumps(record).encode("utf-8"),
        file_options={"content-type": "application/json", "upsert": "true"},
    )
    return diagnosis_id


def _load_diagnoses_from_storage_history(user_id: str) -> List[DiagnosisHistoryItem]:
    """Load user history from Storage JSON records."""
    supabase = get_supabase_client()
    history_items: List[DiagnosisHistoryItem] = []
    folder_path = f"history/{user_id}"

    entries = supabase.storage.from_(IMAGES_BUCKET).list(path=folder_path)
    if not isinstance(entries, list):
        return history_items

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        filename = entry.get("name") or ""
        if not filename.endswith(".json"):
            continue

        record_path = f"{folder_path}/{filename}"
        try:
            raw_data = supabase.storage.from_(IMAGES_BUCKET).download(record_path)
            if isinstance(raw_data, bytes):
                data_str = raw_data.decode("utf-8")
            else:
                data_str = str(raw_data)
            record = json.loads(data_str)
            history_items.append(_build_history_item_from_dict(record))
        except Exception as parse_err:
            logger.warning(f"[HISTORY] ⚠️ Failed to parse storage record {record_path}: {parse_err}")

    history_items.sort(key=lambda item: item.created_at, reverse=True)
    return history_items


def _delete_diagnosis_from_storage_history(user_id: str, diagnosis_id: str) -> bool:
    """Delete a diagnosis JSON record and associated image from Storage fallback."""
    supabase = get_supabase_client()
    record_path = f"history/{user_id}/{diagnosis_id}.json"

    try:
        raw_data = supabase.storage.from_(IMAGES_BUCKET).download(record_path)
        if isinstance(raw_data, bytes):
            data_str = raw_data.decode("utf-8")
        else:
            data_str = str(raw_data)
        record = json.loads(data_str)

        original_image_url = record.get("original_image_url") or ""
        image_path = _extract_storage_object_path(original_image_url)
        if image_path:
            try:
                supabase.storage.from_(IMAGES_BUCKET).remove([image_path])
            except Exception as image_err:
                logger.warning(f"[DELETE] ⚠️ Storage fallback image delete failed: {image_err}")

        supabase.storage.from_(IMAGES_BUCKET).remove([record_path])
        return True
    except Exception as err:
        logger.warning(f"[DELETE] ⚠️ Storage fallback record delete failed: {err}")
        return False


def _clear_storage_history(user_id: str) -> int:
    """Clear all history JSON records (and associated images) for a user."""
    supabase = get_supabase_client()
    folder_path = f"history/{user_id}"
    entries = supabase.storage.from_(IMAGES_BUCKET).list(path=folder_path)
    if not isinstance(entries, list):
        return 0

    deleted_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("name") or ""
        if not filename.endswith(".json"):
            continue

        record_path = f"{folder_path}/{filename}"
        try:
            raw_data = supabase.storage.from_(IMAGES_BUCKET).download(record_path)
            if isinstance(raw_data, bytes):
                data_str = raw_data.decode("utf-8")
            else:
                data_str = str(raw_data)
            record = json.loads(data_str)

            original_image_url = record.get("original_image_url") or ""
            image_path = _extract_storage_object_path(original_image_url)
            if image_path:
                try:
                    supabase.storage.from_(IMAGES_BUCKET).remove([image_path])
                except Exception as image_err:
                    logger.warning(f"[CLEAR_ALL] ⚠️ Storage fallback image delete failed: {image_err}")

            supabase.storage.from_(IMAGES_BUCKET).remove([record_path])
            deleted_count += 1
        except Exception as record_err:
            logger.warning(f"[CLEAR_ALL] ⚠️ Failed to clear storage record {record_path}: {record_err}")

    return deleted_count


@router.post("/predict", response_model=DiagnosisResponse)
async def predict_diagnosis(
    request: Request,
    req: DiagnosisRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Predict disease classification from image.

    Supports:
    - Supabase storage: Provide image_file_path (e.g., "user123/upload_id/image.jpg")
    - External URL: Provide image_url

    Returns:
    - Disease classification (multi-class)
    - Severity (Normal vs Abnormal)
    - Stage (if abnormal)
    - diagnosis_id: ID if saved to database
    """
    try:
        logger.info("[DIAGNOSIS] Starting prediction request")

        # Extract user ID from auth token (optional)
        user_id = None
        if credentials:
            try:
                token = credentials.credentials
                supabase = get_supabase_client()
                user_response = supabase.auth.get_user(token)
                if user_response and user_response.user:
                    user_id = user_response.user.id
                    logger.info(f"[DIAGNOSIS] ✅ Authenticated user: {user_id}")
            except Exception as auth_err:
                logger.warning(f"[DIAGNOSIS] ⚠️ Authentication failed (continuing): {auth_err}")

        # Lazy load model from app state or HuggingFace
        model = None
        if hasattr(request.app.state, 'model'):
            model = request.app.state.model

        if model is None:
            logger.info("[DIAGNOSIS] Model not cached, loading from HuggingFace...")
            try:
                model = get_model()
                request.app.state.model = model
                logger.info("[DIAGNOSIS] ✅ Model loaded and cached")
            except Exception as load_err:
                logger.error(f"[DIAGNOSIS] ❌ Model loading failed: {load_err}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Model initialization failed: {str(load_err)}"
                )

        mapper = get_label_mapper(disease_names=model.disease_names)
        preprocessor = ImagePreprocessor()

        # Get image bytes - try storage path first, fall back to URL
        image_bytes = None

        if req.image_file_path and not req.image_file_path.startswith("http"):
            # Primary: Download via Supabase SDK using storage path
            logger.info("[DIAGNOSIS] Downloading from Supabase storage")
            try:
                image_bytes = await download_from_supabase(req.image_file_path)
            except Exception as storage_err:
                logger.warning(f"[DIAGNOSIS] ⚠️ Supabase storage download failed: {storage_err}")
                # Fall back to URL if available
                if req.image_url:
                    logger.info("[DIAGNOSIS] Falling back to image_url download")
                    image_bytes = await download_from_url(req.image_url)
                else:
                    raise
        elif req.image_url:
            logger.info("[DIAGNOSIS] Downloading from image_url")
            image_bytes = await download_from_url(req.image_url)
        elif req.image_file_path and req.image_file_path.startswith("http"):
            logger.info("[DIAGNOSIS] image_file_path is full URL")
            image_bytes = await download_from_url(req.image_file_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either image_file_path or image_url must be provided"
            )

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to load image"
            )

        # ✅ FIX: Extract multiple tiles matching Kaggle inference exactly
        logger.info("[DIAGNOSIS] Extracting tiles from image...")
        tiles = prepare_tiles(image_bytes, preprocessor)
        logger.info(f"[DIAGNOSIS] Prepared {tiles[0].shape[0]} tiles for inference")

        # Run inference
        logger.info("[DIAGNOSIS] Running inference...")
        prediction = model.predict(tiles)

        logger.info(f"[DIAGNOSIS] 🔬 Raw prediction:")
        logger.info(f"  Disease: {prediction['disease_name']} (conf: {prediction['disease_confidence']:.4f})")
        logger.info(f"  Severity: {prediction['severity_name']} (conf: {prediction['severity_confidence']:.4f})")
        logger.info(f"  Stage: {prediction['stage_name']}")

        # Map to readable labels
        mapped_prediction = mapper.map_prediction(prediction)

        logger.info(f"[DIAGNOSIS] 📋 Mapped prediction:")
        logger.info(f"  Disease: {mapped_prediction['disease']['name']}")
        logger.info(f"  Severity: {mapped_prediction['severity']['name']}")
        logger.info(f"  Diagnosis: {mapped_prediction['diagnosis']}")

        # Build response
        disease = mapped_prediction["disease"]
        severity = mapped_prediction["severity"]
        stage = mapped_prediction["stage"]

        # Determine status based on severity
        diagnosis_status = "abnormal" if severity["level"] == 1 else "normal"

        # Calculate confidence
        confidence_score = (disease["confidence"] + severity["confidence"]) / 2

        if diagnosis_status == "abnormal":
            logger.warning(f"[DIAGNOSIS] ⚠️ Abnormal detected: {mapped_prediction['diagnosis']}")

        # Save to database if authenticated
        diagnosis_id = None
        if user_id:
            try:
                diagnosis_record = Diagnosis(
                    user_id=user_id,
                    disease_name=disease["name"],
                    severity=severity["name"],
                    stage=stage["name"] if stage else None,
                    confidence_disease=disease["confidence"],
                    confidence_severity=severity["confidence"],
                    confidence_stage=stage.get("confidence") if stage else None,
                    original_image_url=req.image_url or req.image_file_path or ""
                )
                db.add(diagnosis_record)
                db.commit()
                db.refresh(diagnosis_record)
                diagnosis_id = str(diagnosis_record.id)
                logger.info(f"[DIAGNOSIS] ✅ Saved to database: {diagnosis_id}")
            except Exception as db_err:
                logger.warning(f"[DIAGNOSIS] ⚠️ SQL save failed, trying Supabase REST fallback: {db_err}")
                db.rollback()
                fallback_payload = {
                    "user_id": user_id,
                    "disease_name": disease["name"],
                    "severity": severity["name"],
                    "stage": stage["name"] if stage else None,
                    "confidence_disease": disease["confidence"],
                    "confidence_severity": severity["confidence"],
                    "confidence_stage": stage.get("confidence") if stage else None,
                    "original_image_url": req.image_url or req.image_file_path or "",
                }
                try:
                    diagnosis_id = _save_diagnosis_via_supabase(
                        user_id=user_id,
                        payload=fallback_payload,
                    )
                    if diagnosis_id:
                        logger.info(f"[DIAGNOSIS] ✅ Saved via Supabase REST fallback: {diagnosis_id}")
                except Exception as supa_err:
                    logger.warning(f"[DIAGNOSIS] ⚠️ Supabase REST fallback failed, trying Storage fallback: {supa_err}")
                    try:
                        diagnosis_id = _save_diagnosis_to_storage_history(
                            user_id=user_id,
                            payload=fallback_payload,
                        )
                        if diagnosis_id:
                            logger.info(f"[DIAGNOSIS] ✅ Saved via Storage fallback: {diagnosis_id}")
                    except Exception as storage_err:
                        logger.warning(f"[DIAGNOSIS] ⚠️ Storage fallback also failed: {storage_err}")

        response = DiagnosisResponse(
            disease=DiseaseInfo(**disease),
            severity=SeverityInfo(**severity),
            stage=StageInfo(**stage) if (stage and severity["level"] == 1) else None,
            region=disease["name"],
            status=diagnosis_status,
            diagnosis=mapped_prediction["diagnosis"],
            confidence_score=confidence_score,
            diagnosis_id=diagnosis_id
        )

        logger.info(f"[DIAGNOSIS] ✅ Diagnosis complete: {response.diagnosis}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DIAGNOSIS] ❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis failed: {str(e)}"
        )


@router.post("/batch")
async def batch_predict_diagnosis(
    request: Request,
    files: List[UploadFile] = File(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Batch diagnosis prediction from multiple uploaded images.
    """
    try:
        logger.info(f"[BATCH] Starting batch prediction for {len(files)} files")

        # Lazy load model
        model = None
        if hasattr(request.app.state, 'model'):
            model = request.app.state.model

        if model is None:
            logger.info("[BATCH] Loading model...")
            try:
                model = get_model()
                request.app.state.model = model
                logger.info("[BATCH] ✅ Model loaded")
            except Exception as load_err:
                logger.error(f"[BATCH] ❌ Model loading failed: {load_err}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Model initialization failed: {str(load_err)}"
                )

        mapper = get_label_mapper(disease_names=model.disease_names)
        preprocessor = ImagePreprocessor()

        results = []

        for file in files:
            try:
                logger.info(f"[BATCH] Processing: {file.filename}")

                # Read file
                image_bytes = await file.read()

                # ✅ FIX: Extract multiple tiles matching Kaggle inference exactly
                tiles = prepare_tiles(image_bytes, preprocessor)
                logger.info(f"[BATCH] Prepared {tiles[0].shape[0]} tiles for {file.filename}")

                # Predict
                prediction = model.predict(tiles)
                mapped_prediction = mapper.map_prediction(prediction)

                # Build result
                disease = mapped_prediction["disease"]
                severity = mapped_prediction["severity"]
                confidence_score = (disease["confidence"] + severity["confidence"]) / 2
                diagnosis_status = "abnormal" if severity["level"] == 1 else "normal"

                if diagnosis_status == "abnormal":
                    logger.warning(f"[BATCH] ⚠️ Abnormal in {file.filename}")

                result = {
                    "filename": file.filename,
                    "disease": disease,
                    "severity": severity,
                    "stage": mapped_prediction["stage"],
                    "region": disease["name"],
                    "status": diagnosis_status,
                    "diagnosis": mapped_prediction["diagnosis"],
                    "confidence_score": confidence_score
                }
                results.append(result)

            except Exception as e:
                logger.warning(f"[BATCH] ⚠️ Failed for {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })

        logger.info(f"[BATCH] ✅ Batch complete: {len(results)} results")
        return {
            "results": results,
            "total": len(files),
            "successful": len([r for r in results if "error" not in r])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[BATCH] ❌ Batch error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch diagnosis failed: {str(e)}"
        )


@router.post("/upload-and-predict")
async def upload_and_predict_diagnosis(
    request: Request,
    file: UploadFile = File(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Upload image and get diagnosis in one request.
    """
    try:
        logger.info(f"[UPLOAD_PREDICT] Processing {file.filename}")

        # Lazy load model
        model = None
        if hasattr(request.app.state, 'model'):
            model = request.app.state.model

        if model is None:
            logger.info("[UPLOAD_PREDICT] Loading model...")
            try:
                model = get_model()
                request.app.state.model = model
                logger.info("[UPLOAD_PREDICT] ✅ Model loaded")
            except Exception as load_err:
                logger.error(f"[UPLOAD_PREDICT] ❌ Model loading failed: {load_err}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Model initialization failed: {str(load_err)}"
                )

        mapper = get_label_mapper(disease_names=model.disease_names)
        preprocessor = ImagePreprocessor()

        # Read uploaded file
        image_bytes = await file.read()

        # ✅ FIX: Extract multiple tiles matching Kaggle inference exactly
        logger.info("[UPLOAD_PREDICT] Extracting tiles from image...")
        tiles = prepare_tiles(image_bytes, preprocessor)
        logger.info(f"[UPLOAD_PREDICT] Prepared {tiles[0].shape[0]} tiles for inference")

        # Predict
        logger.info("[UPLOAD_PREDICT] Running inference...")
        prediction = model.predict(tiles)

        # Map to readable labels
        mapped_prediction = mapper.map_prediction(prediction)

        # Build response
        disease = mapped_prediction["disease"]
        severity = mapped_prediction["severity"]
        stage = mapped_prediction["stage"]

        diagnosis_status = "abnormal" if severity["level"] == 1 else "normal"
        confidence_score = (disease["confidence"] + severity["confidence"]) / 2

        if diagnosis_status == "abnormal":
            logger.warning(f"[UPLOAD_PREDICT] ⚠️ Abnormal: {mapped_prediction['diagnosis']}")

        response = DiagnosisResponse(
            disease=DiseaseInfo(**disease),
            severity=SeverityInfo(**severity),
            stage=StageInfo(**stage) if (stage and severity["level"] == 1) else None,
            region=disease["name"],
            status=diagnosis_status,
            diagnosis=mapped_prediction["diagnosis"],
            confidence_score=confidence_score
        )

        logger.info(f"[UPLOAD_PREDICT] ✅ Complete: {response.diagnosis}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPLOAD_PREDICT] ❌ Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis failed: {str(e)}"
        )


@router.get("/health")
async def diagnosis_health(request: Request):
    """Health check for diagnosis endpoint."""
    try:
        logger.info("[HEALTH] Checking diagnosis service...")

        # Check if model is loaded
        model = None
        if hasattr(request.app.state, 'model'):
            model = request.app.state.model

        if model is None:
            logger.warning("[HEALTH] Model not in cache, attempting to load...")
            try:
                model = get_model()
                request.app.state.model = model
                logger.info("[HEALTH] ✅ Model loaded")
            except Exception as e:
                logger.error(f"[HEALTH] ❌ Model loading failed: {e}")
                return {
                    "status": "unhealthy",
                    "detail": f"Model initialization failed: {str(e)}"
                }

        logger.info("[HEALTH] ✅ Service healthy")
        return {
            "status": "healthy",
            "model": "phase3_mil",
            "diseases": getattr(model, 'disease_names', []),
            "num_diseases": getattr(model, 'num_diseases', 0),
            "stages_supported": getattr(model, 'num_stage_classes', 0) > 0,
            "tile_size": TILE_SIZE,
            "max_tiles": MAX_TILES
        }

    except Exception as e:
        logger.error(f"[HEALTH] ❌ Health check error: {e}")
        return {
            "status": "unhealthy",
            "detail": f"Error: {str(e)}"
        }


# ===== HISTORY ENDPOINTS =====

@router.get("/history", response_model=DiagnosisHistoryResponse)
async def get_diagnosis_history(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Get user's diagnosis history.

    Returns all past diagnoses for the authenticated user.
    """
    try:
        logger.info("[HISTORY] Fetching diagnosis history")

        # Extract user ID from token
        token = credentials.credentials
        supabase = get_supabase_client()
        
        try:
            user_response = supabase.auth.get_user(token)
            if not user_response or not user_response.user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            user_id = user_response.user.id
        except Exception as auth_err:
            logger.error(f"[HISTORY] ❌ Auth error: {auth_err}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        logger.info(f"[HISTORY] User: {user_id}")

        history_items: List[DiagnosisHistoryItem] = []

        try:
            # Primary path: SQLAlchemy DB
            diagnoses = db.query(Diagnosis).filter(
                Diagnosis.user_id == user_id
            ).order_by(Diagnosis.created_at.desc()).all()
            history_items = [_build_history_item_from_record(diagnosis) for diagnosis in diagnoses]
        except Exception as db_err:
            logger.warning(f"[HISTORY] ⚠️ SQL history query failed, trying Supabase REST fallback: {db_err}")
            try:
                supabase_rows = supabase.table("diagnoses").select(
                    "id,disease_name,severity,stage,confidence_disease,confidence_severity,confidence_stage,original_image_url,created_at"
                ).eq("user_id", user_id).order("created_at", desc=True).execute()
                rows = getattr(supabase_rows, "data", None) or []
                history_items = [_build_history_item_from_dict(row) for row in rows]
            except Exception as rest_err:
                logger.warning(f"[HISTORY] ⚠️ Supabase REST history fallback failed, trying Storage fallback: {rest_err}")
                history_items = _load_diagnoses_from_storage_history(str(user_id))

        logger.info(f"[HISTORY] ✅ Retrieved {len(history_items)} diagnoses")

        return DiagnosisHistoryResponse(
            total_diagnoses=len(history_items),
            diagnoses=history_items
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[HISTORY] ❌ Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.delete("/history/{diagnosis_id}")
async def delete_diagnosis(
    diagnosis_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Delete a specific diagnosis and associated images.
    """
    try:
        logger.info(f"[DELETE] Deleting diagnosis: {diagnosis_id}")

        # Extract user ID from token
        token = credentials.credentials
        supabase = get_supabase_client()
        
        try:
            user_response = supabase.auth.get_user(token)
            if not user_response or not user_response.user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            user_id = user_response.user.id
        except Exception as auth_err:
            logger.error(f"[DELETE] ❌ Auth error: {auth_err}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        logger.info(f"[DELETE] User: {user_id}")

        deleted = False

        try:
            # Primary path: SQLAlchemy DB
            diagnosis = db.query(Diagnosis).filter(
                Diagnosis.id == diagnosis_id
            ).first()

            if not diagnosis:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Diagnosis not found: {diagnosis_id}"
                )

            # Verify ownership
            if str(diagnosis.user_id) != str(user_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to delete this diagnosis"
                )

            # Delete images from Supabase if URL exists
            if diagnosis.original_image_url:
                try:
                    file_path = diagnosis.original_image_url
                    if file_path.startswith("http") and "/cytosight-images/" in file_path:
                        file_path = file_path.split("/cytosight-images/")[1].split("?")[0]

                    if file_path:
                        supabase.storage.from_(IMAGES_BUCKET).remove([file_path])
                        logger.info(f"[DELETE] ✅ Deleted image: {file_path}")
                except Exception as del_err:
                    logger.warning(f"[DELETE] ⚠️ Failed to delete image: {del_err}")

            db.delete(diagnosis)
            db.commit()
            deleted = True
        except HTTPException:
            raise
        except Exception as db_err:
            logger.warning(f"[DELETE] ⚠️ SQL delete failed, trying Supabase REST fallback: {db_err}")
            db.rollback()
            try:
                # Fallback path 1: Supabase REST table
                lookup = supabase.table("diagnoses").select("id,user_id,original_image_url").eq("id", diagnosis_id).limit(1).execute()
                rows = getattr(lookup, "data", None) or []
                if not rows:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Diagnosis not found: {diagnosis_id}"
                    )

                diagnosis_row = rows[0]
                if str(diagnosis_row.get("user_id")) != str(user_id):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You don't have permission to delete this diagnosis"
                    )

                original_image_url = diagnosis_row.get("original_image_url") or ""
                if original_image_url:
                    try:
                        file_path = _extract_storage_object_path(original_image_url)
                        if file_path:
                            supabase.storage.from_(IMAGES_BUCKET).remove([file_path])
                    except Exception as del_err:
                        logger.warning(f"[DELETE] ⚠️ Fallback image delete failed: {del_err}")

                supabase.table("diagnoses").delete().eq("id", diagnosis_id).eq("user_id", user_id).execute()
                deleted = True
            except HTTPException:
                raise
            except Exception as rest_err:
                logger.warning(f"[DELETE] ⚠️ Supabase REST delete fallback failed, trying Storage fallback: {rest_err}")
                deleted = _delete_diagnosis_from_storage_history(str(user_id), diagnosis_id)
                if not deleted:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Diagnosis not found: {diagnosis_id}"
                    )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Diagnosis delete did not complete"
            )

        logger.info(f"[DELETE] ✅ Diagnosis deleted: {diagnosis_id}")

        return {
            "success": True,
            "message": f"Diagnosis {diagnosis_id} deleted successfully",
            "diagnosis_id": diagnosis_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DELETE] ❌ Error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete diagnosis: {str(e)}"
        )


@router.delete("/history")
async def clear_all_history(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Delete ALL diagnoses for the user.
    ⚠️ WARNING: This is irreversible!
    """
    try:
        logger.warning("[CLEAR_ALL] Clearing ALL diagnosis history")

        # Extract user ID from token
        token = credentials.credentials
        supabase = get_supabase_client()
        
        try:
            user_response = supabase.auth.get_user(token)
            if not user_response or not user_response.user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            user_id = user_response.user.id
        except Exception as auth_err:
            logger.error(f"[CLEAR_ALL] ❌ Auth error: {auth_err}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        logger.warning(f"[CLEAR_ALL] User: {user_id}")

        delete_count = 0

        try:
            # Primary path: SQLAlchemy DB
            diagnoses = db.query(Diagnosis).filter(
                Diagnosis.user_id == user_id
            ).all()

            logger.info(f"[CLEAR_ALL] Found {len(diagnoses)} diagnoses to delete")

            for diagnosis in diagnoses:
                try:
                    if diagnosis.original_image_url:
                        file_path = diagnosis.original_image_url
                        if file_path.startswith("http") and "cytosight-images/" in file_path:
                            file_path = file_path.split("cytosight-images/")[-1].split("?")[0]

                        if file_path:
                            supabase.storage.from_(IMAGES_BUCKET).remove([file_path])
                except Exception as del_err:
                    logger.warning(f"[CLEAR_ALL] ⚠️ Error deleting image: {del_err}")

            delete_count = db.query(Diagnosis).filter(
                Diagnosis.user_id == user_id
            ).delete()
            db.commit()
        except Exception as db_err:
            logger.warning(f"[CLEAR_ALL] ⚠️ SQL clear failed, trying Supabase REST fallback: {db_err}")
            db.rollback()
            try:
                fallback_rows_resp = supabase.table("diagnoses").select("id,original_image_url").eq("user_id", user_id).execute()
                fallback_rows = getattr(fallback_rows_resp, "data", None) or []

                for row in fallback_rows:
                    try:
                        original_image_url = row.get("original_image_url") or ""
                        if not original_image_url:
                            continue
                        file_path = _extract_storage_object_path(original_image_url)
                        if file_path:
                            supabase.storage.from_(IMAGES_BUCKET).remove([file_path])
                    except Exception as del_err:
                        logger.warning(f"[CLEAR_ALL] ⚠️ Fallback image delete error: {del_err}")

                supabase.table("diagnoses").delete().eq("user_id", user_id).execute()
                delete_count = len(fallback_rows)
            except Exception as rest_err:
                logger.warning(f"[CLEAR_ALL] ⚠️ Supabase REST clear fallback failed, trying Storage fallback: {rest_err}")
                delete_count = _clear_storage_history(str(user_id))

        logger.warning(f"[CLEAR_ALL] ✅ Deleted {delete_count} diagnoses")

        return {
            "success": True,
            "message": "All diagnosis history cleared",
            "deleted_count": delete_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CLEAR_ALL] ❌ Error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {str(e)}"
        )


# ===== HELPER FUNCTIONS =====

async def download_from_supabase(file_path: str) -> bytes:
    """
    Download image from Supabase Storage.

    Args:
        file_path: Path in storage or full signed URL

    Returns:
        Image bytes
    """
    try:
        logger.info(f"[SUPABASE_DL] Downloading: {file_path[:80]}...")

        # If it's a full URL (signed URL), use httpx
        if file_path.startswith("http"):
            logger.info("[SUPABASE_DL] Detected full URL")
            async with httpx.AsyncClient() as client:
                response = await client.get(file_path, timeout=30.0, follow_redirects=True)
                if response.status_code == 200:
                    logger.info(f"[SUPABASE_DL] ✅ Downloaded {len(response.content)} bytes")
                    return response.content
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"URL returned status {response.status_code}"
                    )

        # If it's a path, use Supabase library with robust normalization.
        clean_path = _extract_storage_object_path(file_path)
        if not clean_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Supabase file path"
            )

        logger.info(f"[SUPABASE_DL] Using path: {clean_path}")

        supabase = get_supabase_client()
        last_err = None
        for attempt in range(3):
            try:
                data = supabase.storage.from_(IMAGES_BUCKET).download(clean_path)
                logger.info(f"[SUPABASE_DL] ✅ Downloaded {len(data)} bytes")
                return data
            except Exception as err:
                last_err = err
                # Newly uploaded objects can be briefly unavailable due to propagation.
                logger.warning(
                    f"[SUPABASE_DL] ⚠️ Download attempt {attempt + 1}/3 failed for {clean_path}: {err}"
                )
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))

        raise last_err if last_err else RuntimeError("Supabase download failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SUPABASE_DL] ❌ Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not download image from Supabase: {str(e)}"
        )


async def download_from_url(url: str) -> bytes:
    """
    Download image from URL.

    Args:
        url: Image URL

    Returns:
        Image bytes
    """
    try:
        logger.info(f"[URL_DL] Downloading from URL: {url[:80]}...")

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0, follow_redirects=True)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not download image (status {response.status_code})"
                )

            logger.info(f"[URL_DL] ✅ Downloaded {len(response.content)} bytes")
            return response.content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[URL_DL] ❌ Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error connecting to image URL: {str(e)}"
        )