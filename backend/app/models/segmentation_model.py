"""
Segmentation model wrapper for VQ-VAE based binary mask generation.
CPU-first implementation with tile-based inference for large images.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from transformers import AutoModel
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# HuggingFace-only model sources
HF_SEGMENTATION_REPO = "kimaan28/VQVAE_Segmentation"
HF_SEGMENTATION_FILENAME = "vqvae_best_model.pth"

SEGMENTATION_MODEL_DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ml_models",
    "segmentation",
    "vqvae_best_model.pth",
)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 1.0,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embedding_avg", self.embedding.weight.data.clone())

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        if self.training and self.decay is not None:
            n = torch.sum(encodings, dim=0)
            self.cluster_size.data.mul_(self.decay).add_(n, alpha=1 - self.decay)

            dw = torch.matmul(encodings.t(), flat_input)
            self.embedding_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            normalized_avg = self.embedding_avg / (self.cluster_size.unsqueeze(1) + self.eps)
            self.embedding.weight.data.copy_(normalized_avg)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VQVAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        num_embeddings: int = 4096,
        commitment_cost: float = 1.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_dim // 4),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, embedding_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_dim // 2),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_dim // 4),
            nn.ConvTranspose2d(hidden_dim // 4, input_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        vq_loss, quantized, perplexity, _ = self.vq(z)
        recon_x = self.decoder(quantized)
        return recon_x, vq_loss, perplexity


class SegmentationPipeline:
    def __init__(self, model_path: str, tile_size: int = 384, embedding_dim: int = 256):
        self.model_path = os.path.abspath(model_path)
        self.tile_size = tile_size
        self.embedding_dim = embedding_dim
        self.device = torch.device("cpu")

        self.model = VQVAE(
            input_channels=3,
            hidden_dim=256,
            embedding_dim=self.embedding_dim,
            num_embeddings=4096,
            commitment_cost=1.0,
        ).to(self.device)

        self.to_tensor = transforms.ToTensor()

    def load(self) -> None:
        """Load model weights from local path or HuggingFace Hub."""
        checkpoint_file = None
        
        # 1. Try local paths first
        local_paths = [
            self.model_path,
            SEGMENTATION_MODEL_DEFAULT_PATH,
            "./ml_models/segmentation/vqvae_best_model.pth",
            "../ml_models/segmentation/vqvae_best_model.pth",
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                checkpoint_file = os.path.abspath(path)
                logger.info("[SEGMENTATION]  Found local model at: %s", checkpoint_file)
                break
        
        # 2. Fallback to HuggingFace
        if not checkpoint_file:
            logger.info("[SEGMENTATION]  Local model not found, downloading from HuggingFace: %s", HF_SEGMENTATION_REPO)
            token = os.getenv("HF_TOKEN")
            
            try:
                checkpoint_file = hf_hub_download(
                    repo_id=HF_SEGMENTATION_REPO,
                    filename=HF_SEGMENTATION_FILENAME,
                    token=token,
                    resume_download=True
                )
                logger.info("[SEGMENTATION]  Downloaded from HuggingFace: %s", checkpoint_file)
            except Exception as e:
                logger.error("[SEGMENTATION]  Failed to download from HuggingFace: %s", e)
                raise FileNotFoundError(
                    f"Segmentation model not found locally or on HuggingFace. "
                    f"Repo: {HF_SEGMENTATION_REPO} | Error: {e}"
                )

        # 3. Load the weights
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(cleaned, strict=False)
        self.model.eval()
        logger.info("[SEGMENTATION]  Model loaded successfully")

    @torch.inference_mode()
    def _reconstruct(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        recon, _, _ = self.model(image_tensor)
        return recon

    def _fill_holes_intelligently(self, binary_mask: np.ndarray) -> np.ndarray:
        """Fill holes inside cells intelligently based on area ratios.
        Direct port from Kaggle code."""
        if np.sum(binary_mask == 0) < 100:
            return binary_mask
        
        filled_mask = binary_mask.copy()
        inverted = cv2.bitwise_not(binary_mask)
        contours, hierarchy = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None or len(contours) == 0:
            return binary_mask
        
        hierarchy = hierarchy[0]
        
        for i in range(len(contours)):
            if hierarchy[i][3] != -1:
                parent_idx = hierarchy[i][3]
                parent_contour = contours[parent_idx]
                hole_contour = contours[i]
                
                parent_area = cv2.contourArea(parent_contour)
                hole_area = cv2.contourArea(hole_contour)
                
                if parent_area > 50 and hole_area > 5:
                    area_ratio = hole_area / parent_area
                    
                    if area_ratio < 0.4 and hole_area < 2000:
                        cv2.drawContours(filled_mask, [hole_contour], -1, 0, -1)
        
        return filled_mask

    def _make_binary_mask(self, recon_tensor: torch.Tensor) -> np.ndarray:
        """Creates a binary mask using Otsu thresholding with hole filling.
        Direct port from Kaggle code: create_binary_mask_from_recon."""
        recon_np = recon_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        recon_np = np.clip(recon_np, 0, 1)
        recon_uint8 = (recon_np * 255).astype(np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(recon_uint8, cv2.COLOR_RGB2GRAY)
        
        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding
        _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # White ratio inversion logic
        white_pixels = np.sum(binary_mask == 255)
        total_pixels = binary_mask.size
        white_ratio = white_pixels / total_pixels
        
        if white_ratio < 0.5:
            binary_mask = cv2.bitwise_not(binary_mask)
        
        # Intelligent hole filling
        if np.sum(binary_mask == 255) > 100:
            filled_mask = self._fill_holes_intelligently(binary_mask)
        else:
            filled_mask = binary_mask
            
        return filled_mask

    def _segment_small(self, image: Image.Image) -> np.ndarray:
        original_size = image.size
        resized = image.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
        tensor = self.to_tensor(resized)
        recon = self._reconstruct(tensor)
        mask = self._make_binary_mask(recon)
        return np.array(Image.fromarray(mask).resize(original_size, Image.Resampling.NEAREST), dtype=np.uint8)

    def _segment_large_tiled(self, image: Image.Image) -> np.ndarray:
        width, height = image.size
        tile_size = self.tile_size

        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        padded_w = tiles_x * tile_size
        padded_h = tiles_y * tile_size

        original_np = np.array(image)
        mean_color = tuple(int(v) for v in np.mean(original_np, axis=(0, 1)))
        padded = Image.new("RGB", (padded_w, padded_h), color=mean_color)
        padded.paste(image, (0, 0))

        stitched = np.zeros((padded_h, padded_w), dtype=np.uint8)

        for y in range(tiles_y):
            for x in range(tiles_x):
                x0 = x * tile_size
                y0 = y * tile_size
                x1 = x0 + tile_size
                y1 = y0 + tile_size

                tile = padded.crop((x0, y0, x1, y1))
                tile_tensor = self.to_tensor(tile)
                recon = self._reconstruct(tile_tensor)
                # Process Segmentation Mask with Otsu thresholding, morphology, and hole filling
                # _make_binary_mask already applies all three: thresholding + denoising + hole filling
                tile_mask = self._make_binary_mask(recon)
                stitched[y0:y1, x0:x1] = tile_mask

        return stitched[:height, :width]

    def _estimate_image_size_bytes(self, image: Image.Image) -> int:
        """Estimate the image file size in bytes by serializing to PNG format.
        This gives a reasonable estimate of uncompressed image data size."""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.tell()
        except Exception as err:
            # Fallback: estimate based on raw pixel data (W * H * 3 channels)
            width, height = image.size
            logger.warning("[SEGMENTATION] Could not estimate image size: %s. Using pixel-based estimate.", err)
            return width * height * 3

    def segment(self, image: Image.Image, file_size_bytes: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Segment image using tiling only if image is > 1.5 MB.
        
        Args:
            image: PIL Image to segment
            file_size_bytes: Optional file size in bytes. If not provided, estimated from image data.
        
        Returns:
            Tuple of (mask array, metadata dict)
        """
        width, height = image.size
        
        # Determine if tiling should be used based on image file size (1.5 MB threshold)
        if file_size_bytes is None:
            file_size_bytes = self._estimate_image_size_bytes(image)
        
        threshold_bytes = int(1.5 * 1024 * 1024)  # 1.5 MB
        use_tiling = file_size_bytes > threshold_bytes

        if use_tiling:
            mask = self._segment_large_tiled(image)
        else:
            mask = self._segment_small(image)

        metadata = {
            "width": width,
            "height": height,
            "tile_size": self.tile_size,
            "tiling_used": use_tiling,
        }
        return mask, metadata

    @staticmethod
    def mask_to_png_bytes(mask: np.ndarray) -> bytes:
        img = Image.fromarray(mask.astype(np.uint8), mode="L")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()


_segmentation_pipeline: SegmentationPipeline | None = None


def get_segmentation_pipeline() -> SegmentationPipeline:
    global _segmentation_pipeline

    if _segmentation_pipeline is None:
        model_path = os.getenv("SEGMENTATION_MODEL_PATH", SEGMENTATION_MODEL_DEFAULT_PATH)
        pipeline = SegmentationPipeline(model_path=model_path, tile_size=384, embedding_dim=256)
        pipeline.load()
        _segmentation_pipeline = pipeline

    return _segmentation_pipeline
