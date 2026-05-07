from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
import torch
import numpy as np
import base64
import cv2
import httpx
import traceback
from PIL import Image
import io

from app.models.disease_model import get_model
from app.utils.preprocessing import ImagePreprocessor, load_image_from_bytes
from app.api.diagnosis import _extract_storage_object_path, _standardize_transform
from app.database.supabase_client import get_supabase_client, IMAGES_BUCKET
from app.services.explainability_service import ExplainabilityService, build_overlay

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explainability"])

class ExplainRequest(BaseModel):
    image_url: str
    diagnosis_data: dict

class ExplainResponse(BaseModel):
    attention_heatmap_base64: str
    attention_bbox_base64: str
    highest_attention_crop_base64: str
    zone_reference_base64: str
    gpt_statement: str

def get_image_bytes(image_url: str) -> bytes:
    try:
        # Check if it's a supabase storage path
        clean_path = _extract_storage_object_path(image_url)
        if clean_path and not image_url.startswith("http"):
            # Download from supabase storage
            supabase = get_supabase_client()
            res = supabase.storage.from_(IMAGES_BUCKET).download(clean_path)
            return res
        
        # Download from URL
        with httpx.Client(timeout=30.0) as client:
            response = client.get(image_url, follow_redirects=True)
            response.raise_for_status()
            return response.content
    except Exception as e:
        logger.error(f"Failed to fetch image: {e}")
        raise ValueError(f"Failed to fetch image for explainability: {str(e)}")

def prepare_explainability_input(image_bytes: bytes, preprocessor: ImagePreprocessor) -> tuple[torch.Tensor, np.ndarray]:
    image = load_image_from_bytes(image_bytes)
    
    # Needs to match exactly the model input format
    raw_tile = _standardize_transform(image)
    tile_tensor = preprocessor.transform(raw_tile)
    
    # Add batch dimension
    preprocessed_image = tile_tensor.unsqueeze(0)
    
    # Convert original to numpy for overlay
    img_array = np.array(raw_tile.convert('RGB'))
    
    return preprocessed_image, img_array

def tensor_to_base64(overlay: np.ndarray) -> str:
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV encoding
    overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

@router.post("", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest):
    logger.info(f"Received explainability request for {request.image_url}")
    
    try:
        # 1. Fetch image
        image_bytes = get_image_bytes(request.image_url)
        
        # 2. Get Model
        wrapper = get_model()
        if not wrapper or not wrapper.model:
            raise HTTPException(status_code=503, detail="Model not loaded")
            
        preprocessor = ImagePreprocessor(target_size=256)
        preprocessed_image, original_array = prepare_explainability_input(image_bytes, preprocessor)
        
        # 3. Generate XAI Maps
        service = ExplainabilityService(wrapper)
        xai_results = service.generate_heatmaps(
            preprocessed_image, 
            original_array,
            request.diagnosis_data
        )
        
        # 4. Extract comprehensive features for GPT
        features = service.generate_comprehensive_features(xai_results['attention_heatmap'], original_array)
        
        # 5. Generate Overlays & Base64 conversions
        attention_overlay = build_overlay(original_array, xai_results['attention_heatmap'])
        
        # Helper for direct image to base64 (for bbox, crop, zone which are already RGB/BGR)
        def image_to_base64(img: np.ndarray) -> str:
            # If it's BGR from cv2, keep it. If RGB, convert to BGR for imencode
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"

        # 6. Generate GPT Statement
        gpt_statement = service.generate_gpt_explanation(features, request.diagnosis_data)
        
        return ExplainResponse(
            attention_heatmap_base64=tensor_to_base64(attention_overlay),
            attention_bbox_base64=image_to_base64(xai_results['bbox_overlay']),
            highest_attention_crop_base64=image_to_base64(xai_results['highest_attention_crop']),
            zone_reference_base64=image_to_base64(xai_results['zone_reference']),
            gpt_statement=gpt_statement
        )
        
    except ValueError as e:
        logger.error(f"Value Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error during explainability: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to generate explainability report")
