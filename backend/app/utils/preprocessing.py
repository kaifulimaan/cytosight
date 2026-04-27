"""
Image preprocessing utilities.
Handles image loading, resizing, normalization for model inference.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
from typing import Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Tile extraction size (matches Kaggle SimpleSlideDataset standardize_transform)
TARGET_SIZE = 256

# ViT backbone input size (matches Kaggle test_transform: transforms.Resize(224))
# CRITICAL: The model was trained with 224x224 input to the ViT backbone,
# NOT 256x256. Using 256 here causes feature space mismatch and random predictions.
MODEL_INPUT_SIZE = 224

# ImageNet normalization statistics (used during model training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing pipeline - matches Kaggle test_transform exactly:
#   transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(...)
preprocess_transform = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# For standardization before model (matches training)
resize_transform = transforms.Resize((TARGET_SIZE, TARGET_SIZE))


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load image from bytes.
    
    Args:
        image_bytes: Image data as bytes
    
    Returns:
        PIL Image in RGB format
    
    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {e}")
        raise ValueError(f"Invalid image data: {e}")


def load_image_from_path(image_path: str) -> Image.Image:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PIL Image in RGB format
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If image cannot be loaded
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load image from {image_path}: {e}")
        raise ValueError(f"Invalid image file: {e}")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess single image for model inference.
    
    Args:
        image: PIL Image in RGB format
    
    Returns:
        Preprocessed tensor of shape (3, 224, 224)
    
    Process:
    1. Resize to 224x224 (matches Kaggle test_transform)
    2. Convert to tensor
    3. Normalize with ImageNet statistics
    """
    try:
        tensor = preprocess_transform(image)
        return tensor
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        raise


def preprocess_images_batch(images: list) -> torch.Tensor:
    """
    Preprocess batch of images (tiles).
    
    Args:
        images: List of PIL Images
    
    Returns:
        Batch tensor of shape (num_images, 3, 256, 256)
    """
    tensors = []
    for img in images:
        try:
            tensor = preprocess_image(img)
            tensors.append(tensor)
        except Exception as e:
            logger.warning(f"Failed to preprocess image in batch: {e}")
            continue
    
    if not tensors:
        raise ValueError("No valid images in batch")
    
    return torch.stack(tensors)


def extract_tiles_from_image(
    image: Image.Image, 
    tile_size: int = 256,
    max_tiles: int = 1000
) -> list:
    """
    Extract tile patches from a large image.
    
    Useful for whole slide images (WSI) or large medical images.
    
    Args:
        image: PIL Image
        tile_size: Size of each tile patch (256x256)
        max_tiles: Maximum number of tiles to extract
    
    Returns:
        List of PIL Images (tiles)
    """
    width, height = image.size
    tiles = []
    
    try:
        # Extract non-overlapping tiles
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                if len(tiles) >= max_tiles:
                    break
                
                # Extract tile with padding if at edges
                right = min(x + tile_size, width)
                bottom = min(y + tile_size, height)
                
                tile = image.crop((x, y, right, bottom))
                
                # Pad if necessary to maintain tile_size
                if tile.size != (tile_size, tile_size):
                    padded_tile = Image.new('RGB', (tile_size, tile_size), color=(0, 0, 0))
                    padded_tile.paste(tile, (0, 0))
                    tile = padded_tile
                
                tiles.append(tile)
            
            if len(tiles) >= max_tiles:
                break
        
        logger.info(f"Extracted {len(tiles)} tiles from image ({width}x{height})")
        return tiles
        
    except Exception as e:
        logger.error(f"Failed to extract tiles: {e}")
        raise


def get_image_info(image: Image.Image) -> dict:
    """Get metadata about an image."""
    return {
        'size': image.size,
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format
    }


class ImagePreprocessor:
    """Image preprocessing pipeline."""
    
    def __init__(self, 
                 target_size: int = MODEL_INPUT_SIZE,
                 mean: list = IMAGENET_MEAN,
                 std: list = IMAGENET_STD):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size for ViT backbone input (224 to match Kaggle test_transform)
            mean: Normalization mean values
            std: Normalization std values
        """
        self.target_size = target_size
        self.mean = mean
        self.std = std
        
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def process(self, image_input: Union[str, bytes, Image.Image]) -> torch.Tensor:
        """
        Process image from various input formats.
        
        Args:
            image_input: Image path (str), image bytes, or PIL Image
        
        Returns:
            Preprocessed tensor (3, 256, 256)
        """
        # Load image if needed
        if isinstance(image_input, str):
            image = load_image_from_path(image_input)
        elif isinstance(image_input, bytes):
            image = load_image_from_bytes(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")
        
        # Preprocess
        return preprocess_image(image)
    
    def process_batch(self, images: list) -> torch.Tensor:
        """
        Process batch of images.
        
        Args:
            images: List of image inputs (paths, bytes, or PIL Images)
        
        Returns:
            Batch tensor (batch_size, 3, 256, 256)
        """
        tensors = []
        for img_input in images:
            try:
                tensor = self.process(img_input)
                tensors.append(tensor)
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                continue
        
        if not tensors:
            raise ValueError("No images could be processed")
        
        return torch.stack(tensors)
