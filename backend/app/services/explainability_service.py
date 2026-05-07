import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import logging
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, zoom, maximum_filter
from sklearn.cluster import DBSCAN
from skimage.feature import graycomatrix, graycoprops
from openai import OpenAI
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# -------------------------------------------
# Grad-CAM Implementation for ViT
# -------------------------------------------
class GradCAM:
    """
    Standard Grad-CAM implementation for Vision Transformers
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad = grad_output[0]
                if grad is not None:
                    self.gradients = grad.detach()
            else:
                if grad_output is not None:
                    self.gradients = grad_output.detach()
        
        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()
    
    def generate_cam(self, class_idx, logits):
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return np.zeros((14, 14))
        
        # Global average pooling on gradients to get weights
        weights = self.gradients.mean(dim=1, keepdim=True)  # [B, 1, D]
        cam = (weights * self.activations).sum(dim=2)  # [B, N]
        
        cam = cam[0] # Remove batch dim
        
        # Remove CLS token (first token in ViT)
        if cam.shape[0] > 1:
            cam = cam[1:]
            
        grid_size = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(grid_size, grid_size)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.cpu().numpy()

class GradCAMVisionTransformer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.target_layer = self._find_target_layer()
    
    def _find_target_layer(self):
        vit_model = self.model.backbone.vit
        # For DINOv2 / Phikon-v2, last layer of encoder
        if hasattr(vit_model, 'encoder') and hasattr(vit_model.encoder, 'layer'):
            return vit_model.encoder.layer[-1]
        return vit_model.encoder.layer[-1]
    
    def generate_heatmap(self, image_tensor, target_class_idx, head_type='disease', disease_name=None):
        gradcam = GradCAM(self.model, self.target_layer)
        
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True
        
        # Forward pass with gradients enabled
        disease_logits, severity_logits_dict, stage_logits, _ = self.model([image_tensor], enable_gradients=True)
        
        if head_type == 'disease':
            logits = disease_logits
        elif head_type == 'severity':
            logits = severity_logits_dict[disease_name]
        elif head_type == 'stage':
            logits = stage_logits
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        
        cam = gradcam.generate_cam(target_class_idx, logits)
        gradcam.remove_hooks()
        
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
        cam_smooth = gaussian_filter(cam_resized, sigma=2)
        
        return cam_smooth

# -------------------------------------------
# Attention Extractor
# -------------------------------------------
def extract_patch_level_attention(model_backbone, preprocessed_image, device):
    try:
        if preprocessed_image.dim() == 3:
            image_batch = preprocessed_image.unsqueeze(0).to(device)
        else:
            image_batch = preprocessed_image.to(device)
            
        with torch.no_grad():
            outputs = model_backbone.vit(pixel_values=image_batch)
            hidden_states = outputs.last_hidden_state
            # Exclude CLS token
            patch_tokens = hidden_states[:, 1:, :]
            # Compute importance as L2 norm
            patch_importance = torch.norm(patch_tokens, p=2, dim=2).squeeze().cpu().numpy()
            
            num_patches = len(patch_importance)
            grid_size = int(np.sqrt(num_patches))
            attention_grid = patch_importance.reshape(grid_size, grid_size)
            return attention_grid
    except Exception as e:
        logger.error(f"Error extracting patch attention: {e}")
        return None

def create_patch_attention_heatmap(patch_attention_grid, target_shape):
    zoom_factors = (target_shape[0] / patch_attention_grid.shape[0],
                   target_shape[1] / patch_attention_grid.shape[1])
    heatmap = zoom(patch_attention_grid, zoom_factors, order=1)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = gaussian_filter(heatmap, sigma=5)
    return heatmap

def _generate_zone_reference_image(figsize=(5, 5)):
    """
    Renders a static square bullseye zone reference diagram.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1A1A2E')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor('#1A1A2E')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Periphery ---
    periphery = mpatches.FancyBboxPatch(
        (0.05, 0.05), 0.90, 0.90,
        boxstyle="square,pad=0",
        linewidth=2.0,
        edgecolor='#2ECC71',
        facecolor='#1E3A2E',
        zorder=1
    )
    ax.add_patch(periphery)

    # --- Mid-region ---
    mid = mpatches.FancyBboxPatch(
        (0.22, 0.22), 0.56, 0.56,
        boxstyle="square,pad=0",
        linewidth=2.0,
        linestyle='--',
        edgecolor='#3498DB',
        facecolor='#1E2A3E',
        zorder=2
    )
    ax.add_patch(mid)

    # --- Center ---
    center = mpatches.FancyBboxPatch(
        (0.38, 0.38), 0.24, 0.24,
        boxstyle="square,pad=0",
        linewidth=2.0,
        edgecolor='#9B59B6',
        facecolor='#2E1E3E',
        zorder=3
    )
    ax.add_patch(center)

    # --- Labels ---
    ax.text(0.50, 0.52, 'Center',
            ha='center', va='center', fontsize=8,
            color='#C39BD3', fontweight='bold', zorder=4)
    ax.text(0.50, 0.46, 'innermost',
            ha='center', va='center', fontsize=6.5,
            color='#9B59B6', zorder=4)

    ax.text(0.50, 0.17, 'Mid-region',
            ha='center', va='center', fontsize=7.5,
            color='#85C1E9', fontweight='bold', zorder=4)

    ax.text(0.50, 0.93, 'Periphery',
            ha='center', va='center', fontsize=7.5,
            color='#82E0AA', fontweight='bold', zorder=4)
    ax.text(0.50, 0.88, '(image frame)',
            ha='center', va='center', fontsize=6.0,
            color='#4A8C5C', zorder=4)

    # --- Title ---
    ax.text(0.50, 0.98, 'Spatial attention zone reference',
            ha='center', va='top', fontsize=7,
            color='#A0A0C0', fontweight='bold', zorder=4)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    plt.close(fig)
    return img_array[:, :, :3]

# -------------------------------------------
# Feature Extractor
# -------------------------------------------
class HeatmapFeatureExtractor:
    def __init__(self, heatmap, original_image):
        self.heatmap = heatmap
        self.original_image = original_image
    
    def get_brightest_region(self) -> dict:
        heatmap = self.heatmap.astype(float)
        H, W = heatmap.shape
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        brightest_idx = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
        y_bright, x_bright = brightest_idx
        intensity_bright = heatmap_norm[y_bright, x_bright]
        
        position_bright = self._get_anatomical_position(y_bright, x_bright, H, W)
        
        # Find secondary hotspots
        secondary = []
        neighborhood_size = max(10, min(H, W) // 20)
        local_max = maximum_filter(heatmap_norm, size=neighborhood_size)
        peaks = (heatmap_norm == local_max) & (heatmap_norm > 0.6 * heatmap_norm.max())
        peak_coords = np.argwhere(peaks)
        for idx in range(min(5, len(peak_coords))):
            y, x = peak_coords[idx]
            if np.sqrt((y - y_bright)**2 + (x - x_bright)**2) < 20: continue
            secondary.append({"position": self._get_anatomical_position(y, x, H, W)})
        
        # Spatial coverage
        center_y, center_x = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((Y - center_y)**2 + (X - center_x)**2)
        max_dist = np.sqrt((H/2)**2 + (W/2)**2)
        
        core_mask = dist_from_center < (max_dist * 0.3)
        mid_mask = (dist_from_center >= max_dist * 0.3) & (dist_from_center < max_dist * 0.7)
        periphery_mask = dist_from_center >= (max_dist * 0.7)
        
        total_attention = np.sum(heatmap_norm)
        core_sum = np.sum(heatmap_norm[core_mask])
        mid_sum = np.sum(heatmap_norm[mid_mask])
        periphery_sum = np.sum(heatmap_norm[periphery_mask])
        
        return {
            "primary_hotspot": {
                "position": position_bright,
                "intensity": float(intensity_bright)
            },
            "hotspot_count": 1 + len(secondary),
            "spatial_coverage": {
                "center_attention": float(core_sum / total_attention * 100) if total_attention > 0 else 0,
                "mid_region_attention": float(mid_sum / total_attention * 100) if total_attention > 0 else 0,
                "periphery_attention": float(periphery_sum / total_attention * 100) if total_attention > 0 else 0
            }
        }
        
    def _get_anatomical_position(self, y, x, H, W):
        y_rel, x_rel = y / H, x / W
        dist_from_center = np.sqrt((y - H/2)**2 + (x - W/2)**2)
        dist_ratio = dist_from_center / np.sqrt((H/2)**2 + (W/2)**2)
        
        vert = "upper" if y_rel < 0.35 else "lower" if y_rel > 0.65 else "mid"
        horiz = "left" if x_rel < 0.35 else "right" if x_rel > 0.65 else "center"
        
        if horiz == "center" and vert == "mid": pos = "center"
        elif horiz == "center": pos = f"{vert}-center"
        elif vert == "mid": pos = f"{horiz}-center"
        else: pos = f"{vert}-{horiz}"
        
        if dist_ratio > 0.75: pos = f"{pos} (periphery)"
        elif dist_ratio < 0.3: pos = f"{pos} (core)"
        return pos

    def get_activation_scatter(self, threshold_ratio=0.6) -> dict:
        heatmap = self.heatmap.astype(float)
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        high_pixels = np.argwhere(heatmap_norm >= threshold_ratio * heatmap_norm.max())
        
        if len(high_pixels) == 0:
            return {"scatter_level": "low", "num_clusters": 0}
            
        clustering = DBSCAN(eps=8, min_samples=20).fit(high_pixels)
        num_clusters = len([lb for lb in np.unique(clustering.labels_) if lb != -1])
        
        if num_clusters <= 1: scatter = "low"
        elif 2 <= num_clusters <= 3: scatter = "medium"
        else: scatter = "high"
        
        return {"scatter_level": scatter, "num_clusters": num_clusters}

    def get_dominant_focus_color(self, threshold_ratio=0.6) -> dict:
        heatmap_norm = (self.heatmap - self.heatmap.min()) / (self.heatmap.max() - self.heatmap.min() + 1e-6)
        mask = (heatmap_norm >= threshold_ratio).astype(np.uint8) * 255
        
        if np.sum(mask) == 0: return {"name": "none", "confidence": 0.0}
        
        focus_pixels = self.original_image[mask == 255]
        if len(focus_pixels) < 10: return {"name": "none", "confidence": 0.0}
        
        Z = np.float32(focus_pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(Z, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        counts = np.bincount(labels.flatten())
        dominant_idx = np.argmax(counts)
        dominant_color = centers[dominant_idx].astype(int)
        
        # Simple color mapping
        r, g, b = dominant_color
        hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv
        
        if s < 30: color_name = "gray/pale"
        elif h < 10 or h > 160: color_name = "red/pink"
        elif 10 <= h < 25: color_name = "orange/brown"
        elif 25 <= h < 40: color_name = "yellow/beige"
        elif 40 <= h < 80: color_name = "green"
        elif 80 <= h < 130: color_name = "blue"
        elif 130 <= h < 160: color_name = "purple"
        else: color_name = "mixed"
        
        return {"name": color_name, "confidence": float(counts[dominant_idx] / len(labels) * 100)}

    def get_texture_analysis(self, threshold_ratio=0.6) -> dict:
        heatmap_norm = (self.heatmap - self.heatmap.min()) / (self.heatmap.max() - self.heatmap.min() + 1e-6)
        mask = (heatmap_norm >= threshold_ratio).astype(np.uint8)
        
        if np.sum(mask) < 100: return {"classification": "insufficient", "scores": {"uniformity": 0, "smoothness": 0, "complexity": 0, "organization": 0}}
        
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        focused_gray = gray[mask == 1]
        
        y_coords, x_coords = np.where(mask == 1)
        region = gray[y_coords.min():y_coords.max(), x_coords.min():x_coords.max()]
        quantized_region = (region / 4).astype(np.uint8)
        
        try:
            glcm = graycomatrix(quantized_region, distances=[1], angles=[0, np.pi/4], levels=64, symmetric=True, normed=True)
            energy = float(graycoprops(glcm, 'energy')[0].mean())
            homogeneity = float(graycoprops(glcm, 'homogeneity')[0].mean())
            
            if energy > 0.3: classification = "uniform and smooth"
            elif homogeneity > 0.7: classification = "regular"
            else: classification = "complex/irregular"
            
            return {
                "classification": classification,
                "scores": {
                    "uniformity": int(energy * 100),
                    "smoothness": int(homogeneity * 100),
                    "complexity": int((1-energy) * 100),
                    "organization": int(homogeneity * 100)
                }
            }
        except:
            return {"classification": "error", "scores": {"uniformity": 50, "smoothness": 50, "complexity": 50, "organization": 50}}

    def get_brightest_region_overlay(self, threshold=0.6, alpha=0.5):
        heatmap_norm = (self.heatmap - self.heatmap.min()) / (self.heatmap.max() - self.heatmap.min() + 1e-6)
        mask = (heatmap_norm >= threshold).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply jet colormap to heatmap
        hm_uint8 = (heatmap_norm * 255).astype(np.uint8)
        hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original
        overlay = cv2.addWeighted(self.original_image, 1 - alpha, hm_colored, alpha, 0)
        
        if contours:
            for cnt in contours:
                if cv2.contourArea(cnt) < 50: continue
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Add padding
                pad = 15
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(overlay.shape[1] - x, w + 2*pad)
                h = min(overlay.shape[0] - y, h + 2*pad)
                
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 4)
                
        return {'overlay_with_bbox': overlay}

    def get_highest_attention_crop(self, threshold_ratio=0.6, padding=10):
        heatmap_norm = (self.heatmap - self.heatmap.min()) / (self.heatmap.max() - self.heatmap.min() + 1e-6)
        
        coords = np.argwhere(heatmap_norm >= threshold_ratio * heatmap_norm.max())
        if len(coords) == 0:
            return {'composite_image': self.original_image.copy()}
            
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        
        h, w = self.original_image.shape[:2]
        y1 = max(0, y1 - padding)
        x1 = max(0, x1 - padding)
        y2 = min(h, y2 + padding)
        x2 = min(w, x2 + padding)
        
        crop = self.original_image[y1:y2, x1:x2]
        if crop.size == 0:
            return {'composite_image': self.original_image.copy()}
            
        crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        return {'composite_image': crop_resized}

# -------------------------------------------
# Explainability Service
# -------------------------------------------
class ExplainabilityService:
    def __init__(self, disease_model_wrapper):
        self.wrapper = disease_model_wrapper
        self.device = self.wrapper.device
        self.model = self.wrapper.model
    
    def generate_heatmaps(self, preprocessed_image, original_array, diagnosis_data: dict) -> dict:
        """
        Generates comprehensive XAI results.
        """
        disease_idx = diagnosis_data.get('disease', {}).get('index', 0)
        
        # 1. Attention Heatmap
        patch_attention = extract_patch_level_attention(self.model.backbone, preprocessed_image, self.device)
        attention_heatmap = create_patch_attention_heatmap(patch_attention, (256, 256))
        
        # 2. Extract Features
        extractor = HeatmapFeatureExtractor(attention_heatmap, original_array)
        
        # 3. Generate Overlays & Crops
        bbox_result = extractor.get_brightest_region_overlay(threshold=0.6)
        crop_result = extractor.get_highest_attention_crop(threshold_ratio=0.6)
        zone_ref = _generate_zone_reference_image()
        
        # 4. GradCAM (kept for GPT explanation but not UI)
        try:
            gradcam_wrapper = GradCAMVisionTransformer(self.model, self.device)
            disease_cam = gradcam_wrapper.generate_heatmap(preprocessed_image.clone(), disease_idx, head_type='disease')
            # For simplicity, we just use disease_cam for now
            union_cam = disease_cam
        except Exception as e:
            logger.warning(f"Failed to generate GradCAM: {e}")
            union_cam = attention_heatmap
            
        return {
            'attention_heatmap': attention_heatmap,
            'bbox_overlay': bbox_result['overlay_with_bbox'],
            'highest_attention_crop': crop_result['composite_image'],
            'zone_reference': zone_ref,
            'gradcam_heatmap': union_cam  # Kept for GPT prompt
        }

    def generate_comprehensive_features(self, attention_heatmap, original_array) -> dict:
        extractor = HeatmapFeatureExtractor(attention_heatmap, original_array)
        bright = extractor.get_brightest_region()
        scatter = extractor.get_activation_scatter()
        color = extractor.get_dominant_focus_color()
        texture = extractor.get_texture_analysis()
        
        return {
            'primary_position': bright['primary_hotspot']['position'],
            'primary_intensity': bright['primary_hotspot']['intensity'],
            'hotspot_count': bright['hotspot_count'],
            'center_attention': bright['spatial_coverage']['center_attention'],
            'mid_attention': bright['spatial_coverage']['mid_region_attention'],
            'periphery_attention': bright['spatial_coverage']['periphery_attention'],
            'scatter_level': scatter['scatter_level'],
            'num_clusters': scatter['num_clusters'],
            'dominant_color': color['name'],
            'color_confidence': color['confidence'],
            'texture_classification': texture['classification'],
            'uniformity': texture['scores']['uniformity'],
            'smoothness': texture['scores']['smoothness'],
            'complexity': texture['scores']['complexity'],
            'organization': texture['scores']['organization']
        }

    def generate_gpt_explanation(self, features: dict, diagnosis_data: dict) -> str:
        """Calls OpenAI for textual explanation using the updated Kaggle-style prompt."""
        from app.config import settings
        
        disease_name = diagnosis_data.get('disease', {}).get('name', 'Unknown')
        disease_conf = diagnosis_data.get('disease', {}).get('confidence', 0.0)
        severity = diagnosis_data.get('status', 'Unknown')
        severity_conf = diagnosis_data.get('severity', {}).get('confidence', 0.0)
        stage = diagnosis_data.get('stage', {}).get('name', 'N/A') if diagnosis_data.get('stage') else 'N/A'
        stage_conf = diagnosis_data.get('stage', {}).get('confidence', 0.0) if diagnosis_data.get('stage') else 0.0

        try:
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
                
            client = OpenAI(api_key=api_key)
            
            prompt = f"""You are an AI explainability assistant helping users understand how a hierarchical medical image classification model made its decision. Convert the following technical analysis into a clear, accessible explanation.

HIERARCHICAL MODEL PREDICTION:
- Region: {disease_name} ({disease_conf:.1%} confidence)
- Status Level: {severity} ({severity_conf:.1%} confidence)
- Stage Level: {stage} ({stage_conf:.1%} confidence)

GRADCAM ANALYSIS:
- Note: Bright/warm regions indicate areas that most strongly influenced the model's prediction.

SPATIAL ATTENTION PATTERN AND VISUAL CHARACTERISTICS:
- Primary Focus: {features['primary_position']} (intensity: {features['primary_intensity']:.2f})
- Attention Hotspots: {features['hotspot_count']}
- Spatial Distribution: Center {features['center_attention']:.1f}%, Mid-region {features['mid_attention']:.1f}%, Periphery {features['periphery_attention']:.1f}%
- Clustering: {features['scatter_level']} scatter level with {features['num_clusters']} clusters
- Dominant Color: {features['dominant_color']} ({features['color_confidence']:.1f}% confidence)
- Texture Pattern: {features['texture_classification']}
- Texture Scores: Uniformity {features['uniformity']}/100, Organization {features['organization']}/100, Complexity {features['complexity']}/100, Smoothness {features['smoothness']}/100

CRITICAL INSTRUCTIONS:
1. Write in clear, accessible language for someone without medical or technical expertise.
2. Ground ALL statements in the provided data - do NOT add medical interpretations or diagnoses.
3. Explain how the two explainability methods (Attention Heatmap and GradCAM) show WHERE the model focused.
4. Describe WHAT visual patterns were detected, not WHY medically.
5. Keep it concise but informative (under 100 words).
6. Structure with natural paragraphs, but ensure you cover what the model decided, where it looked, and visual characteristics.
7. Make it conversational but professional.

Generate a comprehensive explanation covering: what the model decided, where it looked, what the Attention and GradCAM methods revealed, and what visual characteristics were important."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at explaining complex AI model decisions in simple, clear language. You help users understand model behavior without making medical claims. Always use [SECTION NAME] headers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}. Using fallback.")
            return f"The model classified this as '{disease_name}' with status '{severity}'. The primary focus was in the {features['primary_position']} region, showing a {features['scatter_level']} scatter across {features['num_clusters']} clusters. The attention distribution was {features['center_attention']:.1f}% central, {features['mid_attention']:.1f}% mid-region, and {features['periphery_attention']:.1f}% peripheral."

def build_overlay(img_array: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Creates a base64 ready RGB image overlay."""
    img_norm = img_array.astype(np.float32) / 255.0
    hm = heatmap.astype(np.float32)

    h, w = img_norm.shape[:2]
    if hm.shape != (h, w):
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)

    hm_min, hm_max = hm.min(), hm.max()
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)

    # Use OpenCV Jet Colormap
    hm_uint8 = (hm * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = img_norm * 0.5 + hm_colored * 0.5
    return np.clip(overlay, 0, 1)

