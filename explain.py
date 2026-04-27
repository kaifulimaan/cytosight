# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
from scipy.ndimage import gaussian_filter

# -------------------------------------------
# Configuration
# -------------------------------------------
OUTPUT_DIR = "/kaggle/working"
GRADCAM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'gradcam_plus_plus_results')
os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)

print("="*80)
print("HIERARCHICAL TRI-HEAD GRAD-CAM++ CONFIGURATION")
print("="*80)
print(f"Output directory: {GRADCAM_OUTPUT_DIR}")
print("="*80)

# -------------------------------------------
# Class Mappings
# -------------------------------------------
DISEASE_CLASS_MAPPING = {
    0: "Breast_cancer",
    1: "annrbc-anemia_processed",
    2: "colon_processed",
    3: "leukemia_processed",
    4: "lung_processed",
    5: "oral-cancer_processed",
    6: "ovarian-cancer_processed",
    7: "sickle-cell-new_processed",
    8: "thalassemia_processed",
}

SEVERITY_CLASS_MAPPING = {
    0: "Normal",
    1: "Abnormal",
}

# -------------------------------------------
# Grad-CAM++ Implementation for ViT
# -------------------------------------------
# -------------------------------------------
# Grad-CAM++ Implementation for ViT (FIXED for tuple outputs)
# -------------------------------------------
class GradCAM:
    """
    Standard Grad-CAM implementation for Vision Transformers
    Simplified - no second-order gradients, just straightforward CAM
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: Your Phase3 hierarchical model
            target_layer: The layer to hook (typically last transformer block)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.handlers = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        def forward_hook(module, input, output):
            # Handle tuple output (DINOv2 returns tuple)
            if isinstance(output, tuple):
                self.activations = output[0].detach()
                print(f"    🪝 Forward hook: Captured from tuple, shape {output[0].shape}")
            else:
                self.activations = output.detach()
                print(f"    🪝 Forward hook: Captured tensor, shape {output.shape}")
        
        def backward_hook(module, grad_input, grad_output):
            # Handle tuple output in gradients
            if isinstance(grad_output, tuple):
                grad = grad_output[0]
                if grad is not None:
                    self.gradients = grad.detach()
                    print(f"    🪝 Backward hook: Captured from tuple, shape {grad.shape}")
            else:
                if grad_output is not None:
                    self.gradients = grad_output.detach()
                    print(f"    🪝 Backward hook: Captured tensor, shape {grad_output.shape}")
        
        # Register hooks
        self.handlers.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.handlers.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.handlers:
            handle.remove()
    
    def generate_cam(self, class_idx, logits):
        """
        Generate standard Grad-CAM heatmap
        
        Args:
            class_idx: Target class index
            logits: Model output logits
            
        Returns:
            cam: Grad-CAM heatmap (H, W)
        """
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # ========================================
        # 🔍 GRADIENT FLOW DEBUGGING
        # ========================================
        print(f"\n    🔍 GRADIENT FLOW CHECK:")
        print(f"    {'='*60}")
        
        # Check if gradients were captured
        if self.gradients is None:
            print(f"    ❌ CRITICAL: No gradients captured!")
            return np.zeros((14, 14))
        else:
            print(f"    ✅ Gradients captured: {self.gradients.shape}")
            print(f"       Min: {self.gradients.min().item():.6f}, Max: {self.gradients.max().item():.6f}")
            print(f"       Mean: {self.gradients.mean().item():.6f}, Std: {self.gradients.std().item():.6f}")
        
        # Check if activations were captured
        if self.activations is None:
            print(f"    ❌ CRITICAL: No activations captured!")
            return np.zeros((14, 14))
        else:
            print(f"    ✅ Activations captured: {self.activations.shape}")
            print(f"       Min: {self.activations.min().item():.6f}, Max: {self.activations.max().item():.6f}")
        
        print(f"    {'='*60}\n")
        # ========================================
        
        # Standard Grad-CAM computation
        # gradients: [B, N, D]
        # activations: [B, N, D]
        
        # Step 1: Global average pooling on gradients to get weights
        # Take mean across spatial dimension (tokens) for each channel
        weights = self.gradients.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        print(f"    📊 Weights (channel importance):")
        print(f"       Shape: {weights.shape}")
        print(f"       Min: {weights.min().item():.6f}, Max: {weights.max().item():.6f}")
        
        # Step 2: Weighted combination of activation maps
        # weights: [B, 1, D]
        # activations: [B, N, D]
        # Result: [B, N] - one value per token
        cam = (weights * self.activations).sum(dim=2)  # Sum across channels (D)
        
        print(f"    📊 CAM before ReLU:")
        print(f"       Shape: {cam.shape}")
        print(f"       Min: {cam.min().item():.6f}, Max: {cam.max().item():.6f}")
        
        # Step 3: Apply ReLU (only keep positive contributions)
        #cam = F.relu(cam)
        
        print(f"    📊 CAM after ReLU:")
        print(f"       Min: {cam.min().item():.6f}, Max: {cam.max().item():.6f}")
        
        # Step 4: Remove batch dimension
        cam = cam[0]  # [N]
        
        # Step 5: Remove CLS token (first token in ViT)
        if cam.shape[0] > 1:
            cam = cam[1:]
            print(f"    🎯 Removed CLS token, remaining tokens: {cam.shape[0]}")
        
        # Step 6: Reshape to spatial grid
        grid_size = int(np.sqrt(cam.shape[0]))
        print(f"    📐 Grid size: {grid_size}x{grid_size}")
        
        cam = cam.reshape(grid_size, grid_size)
        
        # Step 7: Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            print(f"    ✅ Normalized CAM: min={cam.min().item():.4f}, max={cam.max().item():.4f}")
        else:
            print(f"    ⚠️  WARNING: No variation in CAM (all same value)")
            cam = torch.zeros_like(cam)
        
        print(f"    📊 Final CAM mean: {cam.mean().item():.6f}\n")
        
        return cam.cpu().numpy()


class GradCAMVisionTransformer:
    """
    Wrapper to apply standard Grad-CAM to Vision Transformer models
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Find the last transformer block
        self.target_layer = self._find_target_layer()
        print(f"  🎯 Target layer for Grad-CAM: {self.target_layer}")
    
    def _find_target_layer(self):
        """
        Find the last transformer block in DINOv2/Phikon model
        """
        print("\n🔍 Searching for target layer in DINOv2 architecture...")
        
        try:
            if hasattr(self.model, 'backbone'):
                vit_model = self.model.backbone.vit
                print(f"✅ Found backbone.vit: {type(vit_model).__name__}")
            else:
                raise AttributeError("No backbone found")
            
            if hasattr(vit_model, 'encoder') and hasattr(vit_model.encoder, 'layer'):
                num_layers = len(vit_model.encoder.layer)
                last_layer = vit_model.encoder.layer[-1]
                print(f"✅ Found encoder with {num_layers} layers")
                print(f"✅ Target layer: encoder.layer[-1] (layer {num_layers-1})")
                return last_layer
            else:
                raise AttributeError("No encoder.layer found")
                
        except AttributeError as e:
            print(f"❌ Error: {e}")
            raise ValueError("Could not find DINOv2 encoder layers")
    
    def generate_heatmap(self, image_tensor, target_class_idx, head_type='disease'):
        """
        Generate Grad-CAM heatmap for a specific head
        
        Args:
            image_tensor: Input image tensor [1, 3, H, W]
            target_class_idx: Target class index
            head_type: 'disease', 'severity', or 'stage'
            
        Returns:
            heatmap: Grad-CAM heatmap resized to input image size
        """
        # Create Grad-CAM instance
        gradcam = GradCAM(self.model, self.target_layer)
        
        # Forward pass with gradients enabled
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True
        
        # Get logits based on head type
        disease_logits, severity_logits, stage_logits, _ = self.model([image_tensor], enable_gradients=True)
        
        if head_type == 'disease':
            logits = disease_logits
        elif head_type == 'severity':
            disease_pred_idx = disease_logits.argmax(dim=1).item()
            disease_name = DISEASE_CLASS_MAPPING.get(disease_pred_idx, f"Unknown_{disease_pred_idx}")
            logits = severity_logits[disease_name]
        elif head_type == 'stage':
            logits = stage_logits
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        
        # Generate CAM
        cam = gradcam.generate_cam(target_class_idx, logits)
        
        # Clean up hooks
        gradcam.remove_hooks()
        
        # Resize to match input image size
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # Smooth the heatmap
        cam_smooth = gaussian_filter(cam_resized, sigma=2)
        
        return cam_smooth
# -------------------------------------------
# Helper Functions
# -------------------------------------------
def create_gradcam_overlay(image_array, heatmap, alpha=0.5, colormap='jet'):
    """
    Create a visual overlay of Grad-CAM++ heatmap on original image
    Uses red-yellow colormap like traditional Grad-CAM
    
    Args:
        image_array: Original image as numpy array (H, W, 3)
        heatmap: Grad-CAM++ heatmap (H, W), values in [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: Matplotlib colormap name
        
    Returns:
        Overlayed image as numpy array (H, W, 3) in range [0, 1]
    """
    # Normalize image to [0, 1]
    img_normalized = image_array.astype(np.float32) / 255.0
    
    # Resize heatmap to match image size if needed
    target_h, target_w = img_normalized.shape[:2]
    if heatmap.shape != (target_h, target_w):
        print(f"    📐 Resizing heatmap from {heatmap.shape} to ({target_h}, {target_w})")
        heatmap = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # Ensure heatmap is in [0, 1]
    heatmap = np.clip(heatmap, 0, 1)
    
    # Apply colormap - Fixed for newer matplotlib
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Blend with original image
    overlay = img_normalized * (1 - alpha) + heatmap_colored * alpha
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def load_and_preprocess_image(img_path, target_size=224):
    """Load image and preprocess for both visualization and model input"""
    try:
        if img_path.lower().endswith(('.svs', '.ndpi')):
            slide = openslide.OpenSlide(img_path)
            img_pil = slide.get_thumbnail((target_size, target_size))
            slide.close()
        elif img_path.lower().endswith('.tif'):
            try:
                slide = openslide.OpenSlide(img_path)
                img_pil = slide.get_thumbnail((target_size, target_size))
                slide.close()
            except:
                img_pil = Image.open(img_path).convert('RGB')
                img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
        else:
            img_pil = Image.open(img_path).convert('RGB')
            img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
        
        img_array = np.array(img_pil)
        img_tensor = test_transform(img_pil).unsqueeze(0)
        
        return img_pil, img_array, img_tensor
        
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None, None, None


def extract_stage_label(img_path):
    """
    Extract stage label using strict hierarchy rules.
    
    Valid structures:
    - .../abnormal/test/image.png        -> no stage -> return -1
    - .../abnormal/<stage_name>/test/image.png -> stage exists -> return <stage_name>
    """
    path_parts = img_path.split(os.sep)
    
    try:
        # Find 'test' folder
        test_idx = path_parts.index('test')
        
        # Folder immediately above 'test'
        candidate = path_parts[test_idx - 1]
        
        # If abnormal is directly above test → no stage
        if candidate.lower() == 'abnormal':
            return -1
        
        # Otherwise, this folder is the stage
        return candidate
        
    except (ValueError, IndexError):
        # 'test' not found or malformed path
        return -1


# -------------------------------------------
# Main Tri-Head Grad-CAM++ Analysis
# -------------------------------------------
def run_tri_head_gradcam_plus_plus_analysis(model, device, collected_images):
    """
    Run Grad-CAM++ analysis on disease head (Level 1), severity head (Level 2), and stage head (Level 3)
    Stage head is only analyzed when ground truth is abnormal and stage label is not -1
    Returns comprehensive dictionaries with all analysis results
    """
    print("\n" + "="*80)
    print("🔍 STARTING TRI-HEAD GRAD-CAM++ ANALYSIS")
    print("="*80)
    
    if not collected_images:
        print("❌ No images provided")
        return []
    
    print(f"✅ Processing {len(collected_images)} images\n")
    
    # Initialize Grad-CAM++ wrapper
    gradcam_wrapper = GradCAMVisionTransformer(model, device)

    
    all_results = []
    
    for idx, img_path in enumerate(collected_images):
        print(f"\n[{idx+1}/{len(collected_images)}] Processing: {os.path.basename(img_path)}")
        
        try:
            # Load image
            img_pil, img_array, img_tensor = load_and_preprocess_image(img_path, target_size=224)
            
            if img_tensor is None:
                print(f"  ❌ Failed to load image")
                continue
            
            # Extract metadata
            path_parts = img_path.split(os.sep)
            dataset_name = "Unknown"
            true_label = "unknown"
            stage_label = extract_stage_label(img_path)
            
            for part in path_parts:
                if part in ["ovarian-cancer_processed", "oral-cancer_processed", 
                           "Breast_cancer", "colon_processed", "lung_processed",
                           "annrbc-anemia_processed", "leukemia_processed",
                           "sickle-cell-new_processed", "thalassemia_processed"]:
                    dataset_name = part
                if part in ["normal", "abnormal"]:
                    true_label = part
                    break
            
            # ===== STEP 1: Get predictions from all heads =====
            with torch.no_grad():
                img_tensor_device = img_tensor.to(device)
                disease_logits, severity_logits_dict, stage_logits, _ = model([img_tensor_device])
                
                # Disease prediction (Level 1)
                disease_pred_idx = disease_logits.argmax(dim=1).item()
                disease_probs = F.softmax(disease_logits, dim=1)
                disease_confidence = disease_probs[0, disease_pred_idx].item()
                disease_name = DISEASE_CLASS_MAPPING.get(disease_pred_idx, f"Unknown_{disease_pred_idx}")
                disease_all_probs = disease_probs[0].cpu().numpy()
                
                print(f"  📊 Level 1 (Disease): {disease_name}")
                print(f"     Index: {disease_pred_idx}, Confidence: {disease_confidence:.4f}")
                
                # Severity prediction (Level 2)
                severity_logits = severity_logits_dict[disease_name]
                severity_pred_idx = severity_logits.argmax(dim=1).item()
                severity_probs = F.softmax(severity_logits, dim=1)
                severity_confidence = severity_probs[0, severity_pred_idx].item()
                severity_label_text = SEVERITY_CLASS_MAPPING.get(severity_pred_idx, f"Unknown_{severity_pred_idx}")
                severity_all_probs = severity_probs[0].cpu().numpy()
                
                print(f"  📊 Level 2 (Severity): {severity_label_text}")
                print(f"     Index: {severity_pred_idx}, Confidence: {severity_confidence:.4f}")
                
                # Stage prediction (Level 3) - if available
                stage_pred_idx = None
                stage_confidence = None
                stage_all_probs = None
                
                if stage_logits is not None:
                    stage_pred_idx = stage_logits.argmax(dim=1).item()
                    stage_probs = F.softmax(stage_logits, dim=1)
                    stage_confidence = stage_probs[0, stage_pred_idx].item()
                    stage_all_probs = stage_probs[0].cpu().numpy()
                    
                    print(f"  📊 Level 3 (Stage): Stage {stage_pred_idx}")
                    print(f"     Confidence: {stage_confidence:.4f}")
                    print(f"     Ground Truth Stage: {stage_label}")
            
            # ===== STEP 2: Generate Grad-CAM++ for disease head =====
            print(f"\n  🔥 Generating Grad-CAM++ for Disease Head...")
            disease_heatmap = gradcam_wrapper.generate_heatmap(
                img_tensor.clone(),
                disease_pred_idx,
                head_type='disease'
            )
            
            # Create overlay
            disease_overlay = create_gradcam_overlay(img_array, disease_heatmap, alpha=0.5)
            
            print(f"    ✅ Disease heatmap generated")
            print(f"       Min: {disease_heatmap.min():.4f}, Max: {disease_heatmap.max():.4f}")
            
            # ===== STEP 3: Generate Grad-CAM++ for severity head =====
            print(f"  🔥 Generating Grad-CAM++ for Severity Head...")
            severity_heatmap = gradcam_wrapper.generate_heatmap(
                img_tensor.clone(),
                severity_pred_idx,
                head_type='severity'
            )
            
            # Create overlay
            severity_overlay = create_gradcam_overlay(img_array, severity_heatmap, alpha=0.5)
            
            print(f"    ✅ Severity heatmap generated")
            print(f"       Min: {severity_heatmap.min():.4f}, Max: {severity_heatmap.max():.4f}")
            
            # ===== STEP 4: Generate Grad-CAM++ for stage head (conditional) =====
            stage_heatmap = None
            stage_overlay = None
            include_stage_analysis = False
            
            # Check conditions: abnormal ground truth AND stage label != -1
            if true_label == "abnormal" and stage_label != -1 and stage_logits is not None:
                include_stage_analysis = True
                print(f"  🔥 Generating Grad-CAM++ for Stage Head (GT: abnormal, Stage: {stage_label})...")
                
                stage_heatmap = gradcam_wrapper.generate_heatmap(
                    img_tensor.clone(),
                    stage_pred_idx,
                    head_type='stage'
                )
                
                # Create overlay
                stage_overlay = create_gradcam_overlay(img_array, stage_heatmap, alpha=0.5)
                
                print(f"    ✅ Stage heatmap generated")
                print(f"       Min: {stage_heatmap.min():.4f}, Max: {stage_heatmap.max():.4f}")
            else:
                reason = []
                if true_label != "abnormal":
                    reason.append(f"true_label='{true_label}'")
                if stage_label == -1:
                    reason.append("stage_label=-1")
                if stage_logits is None:
                    reason.append("stage_logits=None")
                print(f"  ⏭️  Skipping Stage Head Analysis ({', '.join(reason)})")
            
            # ===== STEP 5: Create Union Heatmap =====
            if include_stage_analysis:
                # Average of all three heatmaps
                union_heatmap = (disease_heatmap + severity_heatmap + stage_heatmap) / 3.0
                print(f"  📊 Union Heatmap: Average of 3 heads (Disease + Severity + Stage)")
            else:
                # Average of two heatmaps
                union_heatmap = (disease_heatmap + severity_heatmap) / 2.0
                print(f"  📊 Union Heatmap: Average of 2 heads (Disease + Severity)")
            
            union_overlay = create_gradcam_overlay(img_array, union_heatmap, alpha=0.5)
            
            # ===== STEP 6: Calculate statistics =====
            disease_mean_activation = float(disease_heatmap.mean())
            disease_max_activation = float(disease_heatmap.max())
            
            severity_mean_activation = float(severity_heatmap.mean())
            severity_max_activation = float(severity_heatmap.max())
            
            stage_mean_activation = None
            stage_max_activation = None
            if stage_heatmap is not None:
                stage_mean_activation = float(stage_heatmap.mean())
                stage_max_activation = float(stage_heatmap.max())
            
            union_mean_activation = float(union_heatmap.mean())
            union_max_activation = float(union_heatmap.max())
            
            # ===== STEP 7: Compile comprehensive results dictionary =====
            result_dict = {
                # ===== Image Information =====
                'filename': os.path.basename(img_path),
                'full_path': img_path,
                'dataset_name': dataset_name,
                'true_label': true_label,
                'stage_label': stage_label,
                'include_stage_analysis': include_stage_analysis,
                
                # ===== Original Image =====
                'image': img_array,
                
                # ===== Level 1: Disease Head Results =====
                'level1_disease': {
                    'predicted_class': disease_name,
                    'predicted_idx': disease_pred_idx,
                    'confidence': disease_confidence,
                    'all_probabilities': disease_all_probs,
                    'heatmap_raw': disease_heatmap,
                    'heatmap_overlay': disease_overlay,
                    'activation_stats': {
                        'mean': disease_mean_activation,
                        'max': disease_max_activation,
                    }
                },
                
                # ===== Level 2: Severity Head Results =====
                'level2_severity': {
                    'predicted_class': severity_label_text,
                    'predicted_idx': severity_pred_idx,
                    'confidence': severity_confidence,
                    'all_probabilities': severity_all_probs,
                    'heatmap_raw': severity_heatmap,
                    'heatmap_overlay': severity_overlay,
                    'activation_stats': {
                        'mean': severity_mean_activation,
                        'max': severity_max_activation,
                    }
                },
                
                # ===== Level 3: Stage Head Results (conditional) =====
                'level3_stage': {
                    'predicted_idx': stage_pred_idx,
                    'confidence': stage_confidence,
                    'all_probabilities': stage_all_probs,
                    'heatmap_raw': stage_heatmap,
                    'heatmap_overlay': stage_overlay,
                    'activation_stats': {
                        'mean': stage_mean_activation,
                        'max': stage_max_activation,
                    } if stage_heatmap is not None else None
                },
                
                # ===== Union Results =====
                'union': {
                    'heatmap_raw': union_heatmap,
                    'heatmap_overlay': union_overlay,
                    'num_heads_averaged': 3 if include_stage_analysis else 2,
                    'activation_stats': {
                        'mean': union_mean_activation,
                        'max': union_max_activation,
                    }
                },
                
                # ===== Legacy Fields =====
                'disease_heatmap': disease_heatmap,
                'severity_heatmap': severity_heatmap,
                'stage_heatmap': stage_heatmap,
                'disease_pred': disease_name,
                'disease_idx': disease_pred_idx,
                'disease_conf': disease_confidence,
                'severity_pred': severity_label_text,
                'severity_idx': severity_pred_idx,
                'severity_conf': severity_confidence,
                'stage_pred_idx': stage_pred_idx,
                'stage_conf': stage_confidence,
            }
            
            all_results.append(result_dict)
            
            print(f"  ✅ Completed tri-head Grad-CAM++ analysis")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("✅ TRI-HEAD GRAD-CAM++ ANALYSIS COMPLETE")
    print(f"📦 Generated {len(all_results)} comprehensive result dictionaries")
    print("="*80)
    
    return all_results


# -------------------------------------------
# Visualization Function
# -------------------------------------------
def display_tri_head_gradcam_grid(results):
    """
    Display grid: each row = one image with 5 columns (or 4 if no stage analysis)
    [Original | Disease Grad-CAM++ | Severity Grad-CAM++ | Stage Grad-CAM++ (if available) | Union]
    """
    if not results:
        print("No results to display")
        return
    
    num_images = len(results)
    max_cols = 5
    
    # Create figure
    fig, axes = plt.subplots(num_images, max_cols, figsize=(35, 7 * num_images))
    
    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    cmap = plt.cm.jet
    
    for i, result in enumerate(results):
        has_stage = result['include_stage_analysis']
        
        # Column 1: Original Image
        axes[i, 0].imshow(result['image'])
        title_text = (
            f"Original Image {i+1}\n"
            f"Dataset: {result['dataset_name']}\n"
            f"True Label: {result['true_label']}\n"
            f"Stage GT: {result['stage_label']}\n"
            f"File: {result['filename']}"
        )
        axes[i, 0].set_title(title_text, fontsize=10, fontweight='bold', pad=10)
        axes[i, 0].axis('off')
        
        # Column 2: Disease Head Grad-CAM++ (Level 1)
        axes[i, 1].imshow(result['image'])
        
        disease_heatmap = result['level1_disease']['heatmap_raw']
        
        im1 = axes[i, 1].imshow(
            disease_heatmap,
            cmap=cmap,
            alpha=0.5,
            vmin=0,
            vmax=1
        )
        
        cbar1 = plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        cbar1.set_label('Activation', rotation=270, labelpad=15)
        
        disease_title = (
            f"Level 1: Disease Head\n"
            f"Predicted: {result['level1_disease']['predicted_class']}\n"
            f"Confidence: {result['level1_disease']['confidence']:.4f}\n"
            f"Mean Act: {result['level1_disease']['activation_stats']['mean']:.4f}"
        )
        axes[i, 1].set_title(disease_title, fontsize=10, fontweight='bold', pad=10)
        axes[i, 1].axis('off')
        
        # Column 3: Severity Head Grad-CAM++ (Level 2)
        axes[i, 2].imshow(result['image'])
        
        severity_heatmap = result['level2_severity']['heatmap_raw']
        
        im2 = axes[i, 2].imshow(
            severity_heatmap,
            cmap=cmap,
            alpha=0.5,
            vmin=0,
            vmax=1
        )
        
        cbar2 = plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        cbar2.set_label('Activation', rotation=270, labelpad=15)
        
        severity_title = (
            f"Level 2: Severity Head\n"
            f"Predicted: {result['level2_severity']['predicted_class']}\n"
            f"Confidence: {result['level2_severity']['confidence']:.4f}\n"
            f"Mean Act: {result['level2_severity']['activation_stats']['mean']:.4f}"
        )
        axes[i, 2].set_title(severity_title, fontsize=10, fontweight='bold', pad=10)
        axes[i, 2].axis('off')
        
        # Column 4: Stage Head Grad-CAM++ (Level 3) - Conditional
        if has_stage:
            axes[i, 3].imshow(result['image'])
            
            stage_heatmap = result['level3_stage']['heatmap_raw']
            
            im3 = axes[i, 3].imshow(
                stage_heatmap,
                cmap=cmap,
                alpha=0.5,
                vmin=0,
                vmax=1
            )
            
            cbar3 = plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
            cbar3.set_label('Activation', rotation=270, labelpad=15)
            
            stage_title = (
                f"Level 3: Stage Head\n"
                f"Predicted: Stage {result['level3_stage']['predicted_idx']}\n"
                f"Confidence: {result['level3_stage']['confidence']:.4f}\n"
                f"Mean Act: {result['level3_stage']['activation_stats']['mean']:.4f}"
            )
            axes[i, 3].set_title(stage_title, fontsize=10, fontweight='bold', pad=10)
            axes[i, 3].axis('off')
        else:
            # Display placeholder text
            axes[i, 3].text(
                0.5, 0.5,
                "Stage Analysis\nNot Applicable\n\n" +
                (f"Reason: GT={result['true_label']}\n" if result['true_label'] != 'abnormal' else "") +
                (f"Stage={result['stage_label']}" if result['stage_label'] == -1 else ""),
                ha='center', va='center',
                fontsize=12, color='gray',
                transform=axes[i, 3].transAxes
            )
            axes[i, 3].axis('off')
        
        # Column 5: Union Grad-CAM++
        axes[i, 4].imshow(result['image'])
        
        union_heatmap = result['union']['heatmap_raw']
        
        im4 = axes[i, 4].imshow(
            union_heatmap,
            cmap=cmap,
            alpha=0.5,
            vmin=0,
            vmax=1
        )
        
        cbar4 = plt.colorbar(im4, ax=axes[i, 4], fraction=0.046, pad=0.04)
        cbar4.set_label('Activation', rotation=270, labelpad=15)
        
        union_title = (
            f"Union: Combined Grad-CAM++\n"
            f"Averaged {result['union']['num_heads_averaged']} Heads\n"
            f"Disease: {result['level1_disease']['predicted_class']}\n"
            f"Severity: {result['level2_severity']['predicted_class']}"
        )
        if has_stage:
            union_title += f"\nStage: {result['level3_stage']['predicted_idx']}"
        union_title += f"\nMean Act: {result['union']['activation_stats']['mean']:.4f}"
        
        axes[i, 4].set_title(union_title, fontsize=10, fontweight='bold', pad=10)
        axes[i, 4].axis('off')
        
        # Print statistics
        print(f"\n📊 Image {i+1} ({result['filename']}) Statistics:")
        print(f"  Disease Head: {result['level1_disease']['predicted_class']} "
              f"({result['level1_disease']['confidence']:.4f})")
        print(f"    Mean Activation: {result['level1_disease']['activation_stats']['mean']:.4f}, "
              f"Max: {result['level1_disease']['activation_stats']['max']:.4f}")
        
        print(f"  Severity Head: {result['level2_severity']['predicted_class']} "
              f"({result['level2_severity']['confidence']:.4f})")
        print(f"    Mean Activation: {result['level2_severity']['activation_stats']['mean']:.4f}, "
              f"Max: {result['level2_severity']['activation_stats']['max']:.4f}")
        
        if has_stage:
            print(f"  Stage Head: Stage {result['level3_stage']['predicted_idx']} "
                  f"({result['level3_stage']['confidence']:.4f})")
            print(f"    Mean Activation: {result['level3_stage']['activation_stats']['mean']:.4f}, "
                  f"Max: {result['level3_stage']['activation_stats']['max']:.4f}")
        else:
            print(f"  Stage Head: Not analyzed (GT: {result['true_label']}, Stage: {result['stage_label']})")
        
        print(f"  Union Heatmap ({result['union']['num_heads_averaged']} heads):")
        print(f"    Mean Activation: {result['union']['activation_stats']['mean']:.4f}, "
              f"Max: {result['union']['activation_stats']['max']:.4f}")
    
    plt.suptitle(
        'Hierarchical Model - Tri-Head Grad-CAM++ Analysis with Union\n'
        'Level 1: Disease | Level 2: Severity | Level 3: Stage (Conditional) | Union: Combined Analysis\n'
        'Red = High Activation | Blue = Low Activation',
        fontsize=16,
        fontweight='bold',
        y=0.998
    )
    
    plt.tight_layout()
    
    # Save
    grid_save_path = os.path.join(GRADCAM_OUTPUT_DIR, 'tri_head_union_gradcam_plus_plus_analysis.png')
    plt.savefig(grid_save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Grid saved to: {grid_save_path}")
    
    plt.show()


# -------------------------------------------
# Execute Analysis
# -------------------------------------------
print("\n" + "="*80)
print("CHECKING FOR COLLECTED IMAGES")
print("="*80)

try:
    if 'collected_images' in locals() or 'collected_images' in globals():
        print(f"✅ Found collected_images with {len(collected_images)} images\n")
        
        # Run tri-head Grad-CAM++ analysis
        gradcam_results = run_tri_head_gradcam_plus_plus_analysis(
            model,
            device,
            collected_images
        )
        
        # Display results
        if gradcam_results:
            display_tri_head_gradcam_grid(gradcam_results)
            
            # Save summary
            results_summary = []
            for r in gradcam_results:
                summary_row = {
                    'filename': r['filename'],
                    'dataset': r['dataset_name'],
                    'true_label': r['true_label'],
                    'stage_gt': r['stage_label'],
                    'disease_predicted': r['level1_disease']['predicted_class'],
                    'disease_confidence': r['level1_disease']['confidence'],
                    'disease_mean_activation': r['level1_disease']['activation_stats']['mean'],
                    'severity_predicted': r['level2_severity']['predicted_class'],
                    'severity_confidence': r['level2_severity']['confidence'],
                    'severity_mean_activation': r['level2_severity']['activation_stats']['mean'],
                }
                
                if r['include_stage_analysis']:
                    summary_row.update({
                        'stage_predicted': r['level3_stage']['predicted_idx'],
                        'stage_confidence': r['level3_stage']['confidence'],
                        'stage_mean_activation': r['level3_stage']['activation_stats']['mean'],
                    })
                else:
                    summary_row.update({
                        'stage_predicted': 'N/A',
                        'stage_confidence': 'N/A',
                        'stage_mean_activation': 'N/A',
                    })
                
                summary_row['union_heads_averaged'] = r['union']['num_heads_averaged']
                summary_row['union_mean_activation'] = r['union']['activation_stats']['mean']
                
                results_summary.append(summary_row)
            
            summary_df = pd.DataFrame(results_summary)
            summary_path = os.path.join(GRADCAM_OUTPUT_DIR, 'tri_head_union_gradcam_plus_plus_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\n✅ Summary saved to: {summary_path}")
            
            print("\n" + "="*80)
            print("TRI-HEAD GRAD-CAM++ ANALYSIS SUMMARY")
            print("="*80)
            print(summary_df.to_string(index=False))
            print("="*80)
            
            # Print structure of results for reference
            print("\n" + "="*80)
            print("📦 RESULTS STRUCTURE")
            print("="*80)
            print("Each result dictionary contains:")
            print("  - filename, full_path, dataset_name, true_label, stage_label")
            print("  - include_stage_analysis: boolean flag")
            print("  - image: original image array")
            print("  - level1_disease: {")
            print("      predicted_class, predicted_idx, confidence, all_probabilities")
            print("      heatmap_raw, heatmap_overlay, activation_stats")
            print("    }")
            print("  - level2_severity: {")
            print("      predicted_class, predicted_idx, confidence, all_probabilities")
            print("      heatmap_raw, heatmap_overlay, activation_stats")
            print("    }")
            print("  - level3_stage: {")
            print("      predicted_idx, confidence, all_probabilities")
            print("      heatmap_raw (None if not analyzed), heatmap_overlay (None if not analyzed)")
            print("      activation_stats (None if not analyzed)")
            print("    }")
            print("  - union: {")
            print("      heatmap_raw (average of 2 or 3 heads)")
            print("      heatmap_overlay, num_heads_averaged, activation_stats")
            print("    }")
            print("="*80)
            
            print(f"\n✅ gradcam_results variable contains {len(gradcam_results)} dictionaries")
            print("   Use gradcam_results in the next cell for further analysis!")
            
        else:
            print("\n❌ No results generated")
    else:
        print("❌ collected_images not found!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()




import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import zoom, gaussian_filter
import os

# -------------------------------------------
# Configuration
# -------------------------------------------
OUTPUT_DIR="/kaggle/working"
ATTENTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'attention_results')
os.makedirs(ATTENTION_OUTPUT_DIR, exist_ok=True)

print("="*80)
print("ATTENTION VISUALIZATION CONFIGURATION")
print("="*80)
print(f"Output directory: {ATTENTION_OUTPUT_DIR}")
print("="*80)

# -------------------------------------------
# Attention Extraction Wrapper
# -------------------------------------------
class AttentionExtractor(nn.Module):
    """
    Wrapper to extract attention weights from the hierarchical model
    """
    def __init__(self, phase3_model):
        super().__init__()
        self.phase3_model = phase3_model
        self.attention_weights = None
        self.tile_features = None
    
    def forward(self, tiles):
        """
        Extract attention weights and tile features from the model
        """
        # Get model outputs including attention weights
        disease_logits, severity_logits, stage_logits, attention_weights = self.phase3_model(tiles)
        
        # Store attention weights for visualization
        self.attention_weights = attention_weights
        
        return disease_logits, severity_logits, stage_logits, attention_weights

# Create attention extractor
attention_extractor = AttentionExtractor(model).to(device)
attention_extractor.eval()

print("\n✅ Attention extractor created successfully\n")

# -------------------------------------------
# Helper Functions
# -------------------------------------------
def extract_attention_map(model, preprocessed_image, device):
    """
    Extract attention weights from the model for a single image
    Since each image is a single tile, we get attention for that single representation
    
    Args:
        model: AttentionExtractor model
        preprocessed_image: Preprocessed tensor (single image, not tiled)
        device: torch device
    
    Returns:
        attention_weights: numpy array of attention weight (single value for single tile)
        disease_logits, severity_logits, stage_logits: model outputs
    """
    model.eval()
    
    try:
        # Each preprocessed image is already a single tensor of shape (C, H, W)
        # We need to add batch dimension and wrap in list
        if preprocessed_image.dim() == 3:
            # Single image: (C, H, W) -> (1, C, H, W)
            image_batch = preprocessed_image.unsqueeze(0)
        else:
            # Already has batch dimension
            image_batch = preprocessed_image
        
        # Wrap in list as model expects list of tile batches
        # Since we have single image as single tile, this is [1 tile batch]
        tiles_list = [image_batch.to(device)]
        
        with torch.no_grad():
            disease_logits, severity_logits, stage_logits, attention_weights = model(tiles_list)
        
        # Convert attention weights to numpy
        # For single tile, this will be shape (1,) or (1, 1)
        attention_np = attention_weights.squeeze().cpu().numpy()
        
        # Ensure it's at least 1D
        if attention_np.ndim == 0:
            attention_np = np.array([attention_np.item()])
        
        print(f"  Extracted attention weights: shape={attention_np.shape}, value={attention_np}")
        
        return attention_np, disease_logits, severity_logits, stage_logits
        
    except Exception as e:
        print(f"  Error extracting attention: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def create_uniform_attention_heatmap(attention_weight, image_shape):
    """
    Create a uniform attention heatmap for a single tile (entire image)
    Since the whole image is one tile, the attention is uniform across it
    
    Args:
        attention_weight: single attention weight value
        image_shape: tuple (height, width) of image
    
    Returns:
        heatmap: 2D array with uniform attention value
    """
    # Since we have single tile = whole image, create uniform heatmap
    # with the attention weight value
    heatmap = np.full(image_shape, attention_weight, dtype=np.float32)
    
    # Normalize to [0, 1] for visualization
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def extract_patch_level_attention(model_backbone, preprocessed_image, device, patch_size=16):
    """
    Extract patch-level attention from ViT backbone
    ViT processes image as patches, we can visualize their importance
    
    Args:
        model_backbone: ViT backbone model
        preprocessed_image: Preprocessed tensor
        device: torch device
        patch_size: ViT patch size (default 16 for most ViTs)
    
    Returns:
        patch_attention_map: 2D heatmap showing patch-level importance
    """
    try:
        if preprocessed_image.dim() == 3:
            image_batch = preprocessed_image.unsqueeze(0).to(device)
        else:
            image_batch = preprocessed_image.to(device)
        
        with torch.no_grad():
            # Get ViT outputs - last_hidden_state contains all patch embeddings
            outputs = model_backbone.vit(pixel_values=image_batch)
            # Shape: (batch, num_patches + 1, embed_dim)
            # First token is CLS token, rest are patch tokens
            
            hidden_states = outputs.last_hidden_state
            
            # Get patch tokens (exclude CLS token at index 0)
            patch_tokens = hidden_states[:, 1:, :]  # (1, num_patches, embed_dim)
            
            # Compute importance as L2 norm of each patch embedding
            patch_importance = torch.norm(patch_tokens, p=2, dim=2).squeeze().cpu().numpy()
            
            # Calculate grid dimensions
            # For 224x224 image with patch_size=16: 14x14 = 196 patches
            num_patches = len(patch_importance)
            grid_size = int(np.sqrt(num_patches))
            
            # Reshape to 2D grid
            attention_grid = patch_importance.reshape(grid_size, grid_size)
            
            print(f"  Extracted patch-level attention: {grid_size}x{grid_size} patches")
            
            return attention_grid
            
    except Exception as e:
        print(f"  Error extracting patch attention: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_patch_attention_heatmap(patch_attention_grid, target_shape):
    """
    Upsample patch-level attention to image dimensions
    
    Args:
        patch_attention_grid: 2D grid of patch attention values
        target_shape: tuple (height, width) for output
    
    Returns:
        heatmap: upsampled attention heatmap
    """
    # Calculate zoom factors
    zoom_factors = (target_shape[0] / patch_attention_grid.shape[0],
                   target_shape[1] / patch_attention_grid.shape[1])
    
    # Upsample using bilinear interpolation
    heatmap = zoom(patch_attention_grid, zoom_factors, order=1)
    
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Apply smoothing for better visualization
    heatmap = gaussian_filter(heatmap, sigma=5)
    
    return heatmap

def load_image_for_attention(img_path, target_size=768):
    """Load and resize image for attention visualization"""
    try:
        if img_path.lower().endswith(('.svs', '.ndpi')):
            slide = openslide.OpenSlide(img_path)
            img_pil = slide.get_thumbnail((target_size, target_size))
            slide.close()
        elif img_path.lower().endswith('.tif'):
            try:
                slide = openslide.OpenSlide(img_path)
                img_pil = slide.get_thumbnail((target_size, target_size))
                slide.close()
            except:
                img_pil = Image.open(img_path).convert('RGB')
                img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
        else:
            img_pil = Image.open(img_path).convert('RGB')
            img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
        
        img_array = np.array(img_pil)
        return img_pil, img_array
        
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None, None

# -------------------------------------------
# Main Attention Extraction Function
# -------------------------------------------
def run_attention_analysis(attention_model, device, collected_images, processed_images,
                          main_class_mapping, stage_class_mapping):
    """
    Extract and visualize attention weights from the hierarchical model
    Uses patch-level attention from ViT backbone since images are single tiles
    
    Args:
        attention_model: AttentionExtractor model
        device: torch device
        collected_images: list of image paths
        processed_images: list of preprocessed tensors (single images)
        main_class_mapping: dictionary mapping class indices to names
        stage_class_mapping: dictionary mapping stage indices to names
    
    Returns:
        list of results dictionaries
    """
    print("\n" + "="*80)
    print("🎯 STARTING ATTENTION WEIGHT EXTRACTION AND VISUALIZATION")
    print("="*80)
    print("ℹ️  Note: Each image is treated as a single tile")
    print("ℹ️  Using patch-level attention from ViT backbone for visualization")
    
    if not collected_images or not processed_images:
        print("❌ No images or preprocessed data provided")
        return []
    
    print(f"✅ Processing {len(collected_images)} images\n")
    
    all_results = []
    
    for idx, (img_path, preprocessed_image) in enumerate(zip(collected_images, processed_images)):
        print(f"\n[{idx+1}/{len(collected_images)}] Processing: {os.path.basename(img_path)}")
        print(f"  Path: {img_path}")
        print(f"  Image shape: {preprocessed_image.shape}")
        
        try:
            # Load original image for visualization
            img_pil, img_array = load_image_for_attention(img_path, target_size=768)
            
            if img_array is None:
                print(f"  ❌ Failed to load image")
                continue
            
            # Extract MIL-level attention weights (single value for single tile)
            attention_weights, disease_logits, severity_logits, stage_logits = extract_attention_map(
                attention_model, 
                preprocessed_image, 
                device
            )
            
            if attention_weights is None:
                print(f"  ❌ Failed to extract attention")
                continue
            
            # Extract patch-level attention from ViT backbone
            patch_attention = extract_patch_level_attention(
                attention_model.phase3_model.backbone,
                preprocessed_image,
                device
            )
            
            if patch_attention is not None:
                # Create heatmap from patch attention
                attention_heatmap = create_patch_attention_heatmap(
                    patch_attention,
                    img_array.shape[:2]
                )
                print(f"  ✅ Created patch-level attention heatmap")
            else:
                # Fallback: uniform heatmap with MIL attention weight
                attention_heatmap = create_uniform_attention_heatmap(
                    attention_weights[0],
                    img_array.shape[:2]
                )
                print(f"  ℹ️  Using uniform attention heatmap")
            
            # Get predictions
            with torch.no_grad():
                disease_probs = F.softmax(disease_logits, dim=1)
                disease_pred_idx = torch.argmax(disease_probs, dim=1).item()
                disease_confidence = disease_probs[0, disease_pred_idx].item()
                
                predicted_class_name = main_class_mapping.get(
                    disease_pred_idx, 
                    f"Unknown_Class_{disease_pred_idx}"
                )
                
                # Get severity prediction
                if "_normal" in predicted_class_name:
                    predicted_disease = predicted_class_name.replace("_normal", "")
                    severity_label = "Normal"
                elif "_abnormal" in predicted_class_name:
                    predicted_disease = predicted_class_name.replace("_abnormal", "")
                    severity_label = "Abnormal"
                else:
                    predicted_disease = predicted_class_name
                    severity_label = "Unknown"
            
            print(f"  📊 Prediction: {predicted_class_name}")
            print(f"     Confidence: {disease_confidence:.4f}")
            print(f"     MIL Attention Weight: {attention_weights[0]:.4f}")
            
            # Extract dataset and true label from path
            path_parts = img_path.split(os.sep)
            dataset_name = "Unknown"
            true_label = "unknown"
            
            for part in path_parts:
                if "processed" in part or "cancer" in part.lower():
                    dataset_name = part
                if part in ["normal", "abnormal"]:
                    true_label = part
                    break
            
            # Store results
            result = {
                'image': img_array,
                'attention_heatmap': attention_heatmap,
                'mil_attention_weight': attention_weights[0],
                'true_label': true_label,
                'dataset_name': dataset_name,
                'predicted_class': predicted_class_name,
                'predicted_disease': predicted_disease,
                'severity': severity_label,
                'class_idx': disease_pred_idx,
                'confidence': disease_confidence,
                'filename': os.path.basename(img_path),
                'full_path': img_path
            }
            
            all_results.append(result)
            
            # Save individual attention heatmap (raw, for OpenCV processing)
            heatmap_filename = f"attention_heatmap_{idx+1}_{os.path.splitext(os.path.basename(img_path))[0]}.npy"
            heatmap_path = os.path.join(ATTENTION_OUTPUT_DIR, heatmap_filename)
            np.save(heatmap_path, attention_heatmap)
            
            print(f"  ✅ Attention heatmap saved to: {heatmap_filename}")
            print(f"  ✅ Completed analysis")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("✅ ATTENTION EXTRACTION COMPLETE")
    print(f"📁 Results saved to: {ATTENTION_OUTPUT_DIR}")
    print("="*80)
    
    return all_results

# -------------------------------------------
# Visualization Function
# -------------------------------------------
def display_attention_grid(results):
    """
    Display grid with original images and attention heatmap overlays
    """
    if not results:
        print("No results to display")
        return
    
    num_images = len(results)
    
    # Create figure: 3 columns (original, heatmap, overlay)
    fig, axes = plt.subplots(num_images, 3, figsize=(18, 6 * num_images))
    
    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # Use 'jet' colormap for attention (blue to red)
    cmap = plt.cm.jet
    
    for i, result in enumerate(results):
        # Column 1: Original Image
        axes[i, 0].imshow(result['image'])
        title_text = (
            f"Original Image {i+1}\n"
            f"Dataset: {result['dataset_name']}\n"
            f"True Label: {result['true_label']}\n"
            f"File: {result['filename'][:30]}..."
        )
        axes[i, 0].set_title(title_text, fontsize=10, fontweight='bold', pad=10)
        axes[i, 0].axis('off')
        
        # Column 2: Attention Heatmap
        im = axes[i, 1].imshow(result['attention_heatmap'], cmap=cmap)
        cbar = plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        heatmap_title = (
            f"Attention Heatmap {i+1}\n"
            f"Patch-Level Importance\n"
            f"MIL Weight: {result['mil_attention_weight']:.4f}"
        )
        axes[i, 1].set_title(heatmap_title, fontsize=10, fontweight='bold', pad=10)
        axes[i, 1].axis('off')
        
        # Column 3: Overlay
        axes[i, 2].imshow(result['image'])
        axes[i, 2].imshow(result['attention_heatmap'], cmap=cmap, alpha=0.5)
        
        overlay_title = (
            f"Overlay {i+1}\n"
            f"Predicted: {result['predicted_class']}\n"
            f"Confidence: {result['confidence']:.4f}"
        )
        axes[i, 2].set_title(overlay_title, fontsize=10, fontweight='bold', pad=10)
        axes[i, 2].axis('off')
        
        # Print statistics
        high_attention = np.sum(result['attention_heatmap'] > 0.7) / result['attention_heatmap'].size * 100
        medium_attention = np.sum((result['attention_heatmap'] > 0.4) & 
                                 (result['attention_heatmap'] <= 0.7)) / result['attention_heatmap'].size * 100
        low_attention = np.sum(result['attention_heatmap'] <= 0.4) / result['attention_heatmap'].size * 100
        
        print(f"\n📊 Image {i+1} ({result['filename']}) Attention Statistics:")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  MIL Attention Weight: {result['mil_attention_weight']:.4f}")
        print(f"  High attention regions (>0.7): {high_attention:.1f}%")
        print(f"  Medium attention regions (0.4-0.7): {medium_attention:.1f}%")
        print(f"  Low attention regions (<0.4): {low_attention:.1f}%")
    
    plt.suptitle(
        'Hierarchical Model - Patch-Level Attention Visualization\n'
        'Warmer colors (red/yellow) indicate higher attention | Cooler colors (blue) indicate lower attention',
        fontsize=16,
        fontweight='bold',
        y=0.998
    )
    
    plt.tight_layout()
    
    # Save grid
    grid_save_path = os.path.join(ATTENTION_OUTPUT_DIR, 'attention_visualization_grid.png')
    plt.savefig(grid_save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Grid visualization saved to: {grid_save_path}")
    
    plt.show()

# -------------------------------------------
# Execute Attention Analysis
# -------------------------------------------
print("\n" + "="*80)
print("CHECKING FOR COLLECTED AND PROCESSED IMAGES")
print("="*80)

try:
    # Check if required variables exist
    if 'collected_images' in locals() or 'collected_images' in globals():
        if 'processed_images' in locals() or 'processed_images' in globals():
            print(f"✅ Found collected_images: {len(collected_images)} images")
            print(f"✅ Found processed_images: {len(processed_images)} tensors\n")
            
            # Run attention analysis
            attention_results = run_attention_analysis(
                attention_extractor,
                device,
                collected_images,
                processed_images,
                DISEASE_CLASS_MAPPING,
                STAGE_CLASS_MAPPING
            )
            
            # Display results
            if attention_results:
                display_attention_grid(attention_results)
                
                # Save results summary
                results_summary = []
                for r in attention_results:
                    high_attn = np.sum(r['attention_heatmap'] > 0.7) / r['attention_heatmap'].size * 100
                    
                    results_summary.append({
                        'filename': r['filename'],
                        'dataset': r['dataset_name'],
                        'true_label': r['true_label'],
                        'predicted_class': r['predicted_class'],
                        'confidence': r['confidence'],
                        'mil_attention_weight': r['mil_attention_weight'],
                        'high_attention_area_%': high_attn,
                        'max_attention': r['attention_heatmap'].max(),
                        'mean_attention': r['attention_heatmap'].mean()
                    })
                
                summary_df = pd.DataFrame(results_summary)
                summary_path = os.path.join(ATTENTION_OUTPUT_DIR, 'attention_summary.csv')
                summary_df.to_csv(summary_path, index=False)
                print(f"\n✅ Summary saved to: {summary_path}")
                
                print("\n" + "="*80)
                print("ATTENTION ANALYSIS SUMMARY")
                print("="*80)
                print(summary_df.to_string(index=False))
                print("="*80)
                
                # Save attention results for next cell (OpenCV feature extraction)
                print("\n" + "="*80)
                print("ATTENTION RESULTS READY FOR OPENCV PROCESSING")
                print("="*80)
                print(f"✅ Variable 'attention_results' contains {len(attention_results)} results")
                print("✅ Each result includes:")
                print("   - Original image")
                print("   - Patch-level attention heatmap (smoothed)")
                print("   - MIL attention weight")
                print("   - Predictions and metadata")
                print("\n💡 Use 'attention_results' in the next cell for OpenCV feature extraction")
                print("="*80)
            else:
                print("\n❌ No results generated")
        else:
            print("❌ processed_images not found!")
            print("Please run the preprocessing cell first")
    else:
        print("❌ collected_images not found!")
        print("Please run the image collection cell first")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
from scipy.ndimage import maximum_filter
from skimage.feature import graycomatrix, graycoprops
from openai import OpenAI
import time
from scipy.stats import spearmanr

# ================================================================
# HEATMAP FEATURE EXTRACTOR CLASS
# Adapted for Hierarchical Model Pipeline
# ================================================================
class HeatmapFeatureExtractor:
    
    def __init__(self, attention_result):
        """
        attention_result is one entry from attention_results
        """
        self.heatmap = attention_result['attention_heatmap']
        self.original_image = attention_result['image']
        self.prediction_info = {
            'predicted_class': attention_result['predicted_class'],
            'predicted_disease': attention_result['predicted_disease'],
            'severity': attention_result['severity'],
            'confidence': attention_result['confidence'],
            'class_idx': attention_result['class_idx'],
            'mil_attention': attention_result['mil_attention_weight']
        }
        self.true_label = attention_result['true_label']
        self.dataset_name = attention_result['dataset_name']
        self.filename = attention_result['filename']
    
    # ---------------------------------------------------------
    # METHOD 1: Brightest Region Analysis
    # ---------------------------------------------------------
    def get_brightest_region(self):
        """
        IMPROVED: Comprehensive analysis of high-attention regions
        """
        heatmap = self.heatmap.astype(float)
        H, W = heatmap.shape
        
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # 1. PRIMARY HOTSPOT
        brightest_idx = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
        y_bright, x_bright = brightest_idx
        intensity_bright = heatmap_norm[y_bright, x_bright]
        
        position_bright = self._get_anatomical_position(y_bright, x_bright, H, W)
        
        primary_hotspot = {
            "pixel": (int(y_bright), int(x_bright)),
            "position": position_bright,
            "intensity": float(intensity_bright)
        }
        
        # 2. SECONDARY HOTSPOTS
        secondary_hotspots = self._find_secondary_hotspots(heatmap_norm, H, W, threshold=0.6)
        
        # 3. ATTENTION PATTERN
        attention_pattern = self._determine_attention_pattern(heatmap_norm, H, W)
        
        # 4. SPATIAL COVERAGE
        spatial_coverage = self._calculate_spatial_coverage(heatmap_norm, H, W)
        
        # 5. HOTSPOT COUNT
        hotspot_count = 1 + len(secondary_hotspots)
        
        return {
            "primary_hotspot": primary_hotspot,
            "secondary_hotspots": secondary_hotspots,
            "attention_pattern": attention_pattern,
            "spatial_coverage": spatial_coverage,
            "hotspot_count": hotspot_count
        }

    def _get_anatomical_position(self, y, x, H, W):
        """Convert pixel coordinates to descriptive position"""
        y_rel = y / H
        x_rel = x / W
        
        center_y, center_x = H / 2, W / 2
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_dist = np.sqrt((H/2)**2 + (W/2)**2)
        dist_ratio = dist_from_center / max_dist
        
        center_threshold_inner = 0.35
        center_threshold_outer = 0.65
        periphery_threshold = 0.75
    
        if y_rel < center_threshold_inner:
            vert = "upper"
        elif y_rel > center_threshold_outer:
            vert = "lower"
        else:
            vert = "mid"
        
        if x_rel < center_threshold_inner:
            horiz = "left"
        elif x_rel > center_threshold_outer:
            horiz = "right"
        else:
            horiz = "center"
        
        if horiz == "center" and vert == "mid":
            position = "center"
        elif horiz == "center":
            position = f"{vert}-center"
        elif vert == "mid":
            position = f"{horiz}-center"
        else:
            position = f"{vert}-{horiz}"
        
        if dist_ratio > periphery_threshold:
            position = f"{position} (periphery)"
        elif dist_ratio < 0.3:
            position = f"{position} (core)"
        
        return position
    
    def _find_secondary_hotspots(self, heatmap_norm, H, W, threshold=0.6, min_distance=20):
        """Find additional significant attention regions"""
        secondary = []
        
        neighborhood_size = max(10, min(H, W) // 20)
        local_max = maximum_filter(heatmap_norm, size=neighborhood_size)
        
        peaks = (heatmap_norm == local_max) & (heatmap_norm > threshold * heatmap_norm.max())
        peak_coords = np.argwhere(peaks)
        peak_intensities = heatmap_norm[peaks]
        sorted_indices = np.argsort(peak_intensities)[::-1]
        
        primary_y, primary_x = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
        
        for idx in sorted_indices[:5]:
            y, x = peak_coords[idx]
            
            if np.sqrt((y - primary_y)**2 + (x - primary_x)**2) < min_distance:
                continue
            
            too_close = False
            for existing in secondary:
                ey, ex = existing["pixel"]
                if np.sqrt((y - ey)**2 + (x - ex)**2) < min_distance:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            position = self._get_anatomical_position(y, x, H, W)
            intensity = float(heatmap_norm[y, x])
            
            secondary.append({
                "pixel": (int(y), int(x)),
                "position": position,
                "intensity": intensity
            })
        
        return secondary

    def _determine_attention_pattern(self, heatmap_norm, H, W):
        """Determine overall attention distribution pattern"""
        center_y, center_x = H // 2, W // 2
        
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((Y - center_y)**2 + (X - center_x)**2)
        max_dist = np.sqrt((H/2)**2 + (W/2)**2)
        
        core_mask = dist_from_center < (max_dist * 0.3)
        mid_mask = (dist_from_center >= max_dist * 0.3) & (dist_from_center < max_dist * 0.7)
        periphery_mask = dist_from_center >= (max_dist * 0.7)
        
        core_attention = np.mean(heatmap_norm[core_mask])
        mid_attention = np.mean(heatmap_norm[mid_mask])
        periphery_attention = np.mean(heatmap_norm[periphery_mask])
        
        high_attention_pixels = np.sum(heatmap_norm > 0.7) / heatmap_norm.size
        
        if core_attention > 0.7 and core_attention > mid_attention * 1.5:
            return "centralized (focused on center)"
        elif periphery_attention > 0.7 and periphery_attention > core_attention * 1.5:
            return "peripheral (focused on edges)"
        elif mid_attention > core_attention and mid_attention > periphery_attention:
            return "ring-like (donut pattern)"
        elif high_attention_pixels > 0.5:
            return "diffuse (spread across image)"
        elif high_attention_pixels > 0.1 and high_attention_pixels < 0.3:
            return "focal (single concentrated region)"
        else:
            return "scattered (multiple regions)"

    def _calculate_spatial_coverage(self, heatmap_norm, H, W):
        """Calculate percentage of attention in each spatial region"""
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
            "center_attention": float(core_sum / total_attention * 100) if total_attention > 0 else 0,
            "mid_region_attention": float(mid_sum / total_attention * 100) if total_attention > 0 else 0,
            "periphery_attention": float(periphery_sum / total_attention * 100) if total_attention > 0 else 0
        }

    # ---------------------------------------------------------
    # METHOD 2: Scatter Analysis
    # ---------------------------------------------------------
    def get_activation_scatter(self, threshold_ratio=0.6):
        """Determine if heatmap is focused or scattered"""
        heatmap = self.heatmap.astype(float)
        H, W = heatmap.shape

        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        thresh = threshold_ratio * heatmap_norm.max()
        high_pixels = np.argwhere(heatmap_norm >= thresh)

        if len(high_pixels) == 0:
            return {
                "scatter_level": "low",
                "num_clusters": 0,
                "clusters_sizes": []
            }

        clustering = DBSCAN(eps=8, min_samples=20).fit(high_pixels)
        labels = clustering.labels_
        unique_labels = [lb for lb in np.unique(labels) if lb != -1]

        cluster_sizes = []
        for lb in unique_labels:
            cluster_sizes.append(int(np.sum(labels == lb)))

        num_clusters = len(unique_labels)

        if num_clusters == 1:
            scatter = "low"
        elif 2 <= num_clusters <= 3:
            scatter = "medium"
        else:
            scatter = "high"

        return {
            "scatter_level": scatter,
            "num_clusters": num_clusters,
            "clusters_sizes": cluster_sizes
        }
    
    # ---------------------------------------------------------
    # METHOD 3: Dominant Color Analysis
    # ---------------------------------------------------------
    def get_dominant_focus_color(self, threshold_ratio=0.6, k_clusters=5):
        """Detect dominant color in attention-focused regions"""
        heatmap = self.heatmap.astype(float)
        orig = self.original_image.copy()
    
        H, W = heatmap.shape
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        mask = (heatmap_norm >= threshold_ratio).astype(np.uint8) * 255
    
        if np.sum(mask) == 0:
            return {
                "dominant_color_rgb": None,
                "dominant_color_hsv": None,
                "dominant_color_name": "none",
                "color_confidence": 0.0
            }
    
        kernel = np.ones((3, 3), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return {
                "dominant_color_rgb": None,
                "dominant_color_hsv": None,
                "dominant_color_name": "none",
                "color_confidence": 0.0
            }
    
        activation_mask = np.zeros_like(mask_clean)
        cv2.drawContours(activation_mask, contours, -1, 255, -1)
        focus_pixels = orig[activation_mask == 255]
    
        if len(focus_pixels) < 10:
            return {
                "dominant_color_rgb": None,
                "dominant_color_hsv": None,
                "dominant_color_name": "none",
                "color_confidence": 0.0
            }

        Z = np.float32(focus_pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = k_clusters
    
        _, labels, centers = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
    
        counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(counts)[::-1]
        
        dominant_index = None
        dominant_color = None
        
        for idx in sorted_indices:
            candidate_color = centers[idx].astype(int)
            hsv_color = cv2.cvtColor(
                np.uint8([[candidate_color]]), 
                cv2.COLOR_RGB2HSV
            )[0][0]
        
            if hsv_color[1] > 30:
                dominant_index = idx
                dominant_color = candidate_color
                break
        
        if dominant_index is None:
            dominant_index = sorted_indices[0]
            dominant_color = centers[dominant_index].astype(int)
        
        hsv_color = cv2.cvtColor(
            np.uint8([[dominant_color]]), 
            cv2.COLOR_RGB2HSV
        )[0][0]
        
        color_confidence = counts[dominant_index] / len(labels) * 100
        dominant_name = self._map_color_to_name(
            dominant_color.tolist(), 
            hsv_color.tolist()
        )
    
        return {
            "dominant_color_rgb": dominant_color.tolist(),
            "dominant_color_hsv": hsv_color.tolist(),
            "dominant_color_name": dominant_name,
            "color_confidence": float(color_confidence)
        }

    def _map_color_to_name(self, rgb, hsv=None):
        """Enhanced color naming using HSV color space"""
        r, g, b = rgb
    
        if hsv is None:
            hsv_array = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = hsv_array.tolist()
        else:
            h, s, v = hsv
        
        if s < 30:
            if v > 200:
                return "white / very light"
            elif v > 150:
                return "light gray / pale"
            elif v > 80:
                return "gray"
            else:
                return "dark gray / black"
        
        if v < 60:
            return "very dark / black"
    
        if 130 <= h <= 160:
            if s > 100:
                return "purple / violet"
            else:
                return "light purple / lavender"
        
        if 160 <= h <= 180 or h <= 10:
            if v > 180 and s < 100:
                return "pink / light red"
            elif s > 150:
                return "magenta / bright pink"
            else:
                return "pink / rose"
    
        if h <= 10:
            if v < 150:
                return "dark red / maroon"
            else:
                return "red / crimson"
        
        if 10 <= h < 25:
            if v < 130:
                return "brown / dark tan"
            else:
                return "orange / tan"
        
        if 25 <= h < 40:
            if s < 80:
                return "beige / cream"
            else:
                return "yellow / golden"
        
        if 40 <= h < 80:
            if v > 180:
                return "light green / pale green"
            elif v > 120:
                return "green"
            else:
                return "dark green"
    
        if 80 <= h < 100:
            return "cyan / turquoise"
        
        if 100 <= h < 130:
            if s > 150:
                return "blue / deep blue"
            elif v > 180:
                return "light blue / sky blue"
            else:
                return "blue"
        
        max_channel = max(r, g, b)
        if max_channel == r:
            return "reddish tones"
        elif max_channel == g:
            return "greenish tones"
        elif max_channel == b:
            return "bluish tones"
        else:
            return "mixed color region"

    # ---------------------------------------------------------
    # METHOD 4: Texture Analysis
    # ---------------------------------------------------------
    def get_texture_analysis(self, threshold_ratio=0.6):
        """
        Analyze texture patterns in high-attention regions using GLCM
        
        Returns generic image-based descriptions without medical assumptions
        """
        heatmap = self.heatmap.astype(float)
        orig = self.original_image.copy()
        
        H, W = heatmap.shape
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        mask = (heatmap_norm >= threshold_ratio).astype(np.uint8) * 255
        
        if np.sum(mask) < 100:
            return {
                "texture_classification": "insufficient data",
                "texture_description": "Not enough attention data to analyze texture",
                "texture_scores": {
                    "uniformity": 0,
                    "organization": 0,
                    "complexity": 0,
                    "smoothness": 0
                },
                "glcm_features": {
                    "contrast": 0.0,
                    "correlation": 0.0,
                    "energy": 0.0,
                    "homogeneity": 0.0
                }
            }
        
        # Clean mask
        kernel = np.ones((3, 3), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Extract focused region from original image
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                "texture_classification": "no region detected",
                "texture_description": "No focused region detected",
                "texture_scores": {
                    "uniformity": 0,
                    "organization": 0,
                    "complexity": 0,
                    "smoothness": 0
                },
                "glcm_features": {
                    "contrast": 0.0,
                    "correlation": 0.0,
                    "energy": 0.0,
                    "homogeneity": 0.0
                }
            }
        
        # Create activation mask
        activation_mask = np.zeros_like(mask_clean)
        cv2.drawContours(activation_mask, contours, -1, 255, -1)
        
        # Get bounding box of the region
        y_coords, x_coords = np.where(activation_mask == 255)
        if len(y_coords) == 0:
            return {
                "texture_classification": "invalid region",
                "texture_description": "Invalid region for texture analysis",
                "texture_scores": {
                    "uniformity": 0,
                    "organization": 0,
                    "complexity": 0,
                    "smoothness": 0
                },
                "glcm_features": {
                    "contrast": 0.0,
                    "correlation": 0.0,
                    "energy": 0.0,
                    "homogeneity": 0.0
                }
            }
        
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        # Extract region
        region_rgb = orig[y_min:y_max+1, x_min:x_max+1]
        region_mask = activation_mask[y_min:y_max+1, x_min:x_max+1]
        
        # Convert to grayscale for texture analysis
        region_gray = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply mask to focus only on high-attention pixels
        region_gray_masked = region_gray.copy()
        region_gray_masked[region_mask == 0] = 0
        
        # Quantize to reduce GLCM computation (64 levels)
        region_quantized = (region_gray_masked / 4).astype(np.uint8)
        
        # Compute GLCM
        # distances: [1] means immediate neighbors
        # angles: [0, π/4, π/2, 3π/4] for rotation invariance
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        try:
            glcm = graycomatrix(
                region_quantized,
                distances=distances,
                angles=angles,
                levels=64,
                symmetric=True,
                normed=True
            )
            
            # Extract features (averaged across all angles)
            contrast = float(graycoprops(glcm, 'contrast')[0].mean())
            correlation = float(graycoprops(glcm, 'correlation')[0].mean())
            energy = float(graycoprops(glcm, 'energy')[0].mean())
            homogeneity = float(graycoprops(glcm, 'homogeneity')[0].mean())
            
        except Exception as e:
            print(f"  Warning: GLCM computation failed: {e}")
            return {
                "texture_classification": "computation error",
                "texture_description": "Error computing texture features",
                "texture_scores": {
                    "uniformity": 0,
                    "organization": 0,
                    "complexity": 0,
                    "smoothness": 0
                },
                "glcm_features": {
                    "contrast": 0.0,
                    "correlation": 0.0,
                    "energy": 0.0,
                    "homogeneity": 0.0
                }
            }
        
        # Convert to 0-100 scores
        uniformity_score = int(energy * 100)
        organization_score = int(max(0, min(100, (correlation + 1) * 50)))  # Scale -1,1 to 0,100
        complexity_score = int((1 - energy) * 100)
        smoothness_score = int(homogeneity * 100)
        
        # Classify texture based on GLCM features
        classification, description = self._classify_texture(
            contrast, correlation, energy, homogeneity
        )
        
        return {
            "texture_classification": classification,
            "texture_description": description,
            "texture_scores": {
                "uniformity": uniformity_score,
                "organization": organization_score,
                "complexity": complexity_score,
                "smoothness": smoothness_score
            },
            "glcm_features": {
                "contrast": round(contrast, 2),
                "correlation": round(correlation, 3),
                "energy": round(energy, 3),
                "homogeneity": round(homogeneity, 3)
            }
        }
    
    def _classify_texture(self, contrast, correlation, energy, homogeneity):
        """
        Classify texture based on GLCM features
        Returns (classification, description) tuple
        """
        # Rule-based classification
        if contrast < 100 and homogeneity > 0.8:
            classification = "uniform and smooth"
            description = "Model focused on a region with smooth, uniform texture showing consistent patterns with minimal variation"
        
        elif correlation > 0.7 and energy > 0.3:
            classification = "structured and regular"
            description = "Model focused on a region with organized, structured patterns exhibiting regular, repeating elements"
        
        elif contrast > 400 and correlation < 0.4:
            classification = "irregular and chaotic"
            description = "Model focused on a region with irregular, chaotic texture displaying highly variable patterns with no clear organization"
        
        elif homogeneity < 0.5:
            classification = "rough and coarse"
            description = "Model focused on a region with rough, coarse texture showing sharp intensity changes and abrupt transitions"
        
        elif energy < 0.2:
            classification = "complex and varied"
            description = "Model focused on a region with complex, varied texture containing multiple different patterns and high visual diversity"
        
        else:
            classification = "moderate texture"
            description = "Model focused on a region with moderate texture complexity showing intermediate characteristics"
        
        return classification, description
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from transformers import Dinov2Model
import openslide
from tqdm import tqdm
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from openai import OpenAI
import time

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "/kaggle/input/models/ulimaank/updated-diagnostic-model-jan-18/other/default/1/phase3_mil_best.pth"
OUTPUT_DIR = "/kaggle/working"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, GPUs: {torch.cuda.device_count()}")

DISEASE_NAMES = [
    'Breast_cancer', 'annrbc-anemia_processed', 'colon_processed',
    'leukemia_processed', 'lung_processed', 'oral-cancer_processed',
    'ovarian-cancer_processed', 'sickle-cell-new_processed', 'thalassemia_processed'
]

STAGE_NAMES = {
    0: 'Breast_cancer - ductal_carcinoma',
    1: 'Breast_cancer - lobular_carcinoma',
    2: 'Breast_cancer - mucinous_carcinoma',
    3: 'Breast_cancer - papillary_carcinoma',
    4: 'leukemia_processed - Early',
    5: 'leukemia_processed - Pre',
    6: 'leukemia_processed - Pro',
    7: 'lung_processed - lung_aca',
    8: 'lung_processed - lung_scc',
    9: 'ovarian-cancer_processed - CC',
    10: 'ovarian-cancer_processed - EC',
    11: 'ovarian-cancer_processed - HGSC',
    12: 'ovarian-cancer_processed - LGSC',
    13: 'ovarian-cancer_processed - MC'
}

DISEASE_CLASS_MAPPING = {
    0: "Breast_cancer",
    1: "annrbc-anemia_processed",
    2: "colon_processed",
    3: "leukemia_processed",
    4: "lung_processed",
    5: "oral-cancer_processed",
    6: "ovarian-cancer_processed",
    7: "sickle-cell-new_processed",
    8: "thalassemia_processed",
}

STAGE_CLASS_MAPPING = STAGE_NAMES

TARGET_SIZE = 256
standardize_transform = transforms.Resize((TARGET_SIZE, TARGET_SIZE))


# ================================================================
# DATA COLLECTION
# ================================================================

def collect_images_from_folder(folder_path):
    images = []
    valid_extensions = ('.svs', '.tif', '.ndpi', '.png', '.jpg', '.jpeg', '.tiff')
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(valid_extensions):
                images.append(os.path.join(root, f))
    return images


# ================================================================
# DATASET
# ================================================================

class SimpleSlideDataset(Dataset):
    def __init__(self, image_paths, tile_size=224, max_tiles=1000):
        self.image_paths = image_paths
        self.tile_size = tile_size
        self.max_tiles = max_tiles

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        slide_path = self.image_paths[idx]
        tiles = []
        try:
            if slide_path.lower().endswith(('.svs', '.ndpi')):
                slide = openslide.OpenSlide(slide_path)
                width, height = slide.dimensions
                for y in range(0, height, self.tile_size):
                    for x in range(0, width, self.tile_size):
                        if len(tiles) >= self.max_tiles:
                            break
                        tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')
                        tiles.append(standardize_transform(tile))
                    if len(tiles) >= self.max_tiles:
                        break
                slide.close()
            elif slide_path.lower().endswith('.tif'):
                try:
                    slide = openslide.OpenSlide(slide_path)
                    width, height = slide.dimensions
                    for y in range(0, height, self.tile_size):
                        for x in range(0, width, self.tile_size):
                            if len(tiles) >= self.max_tiles:
                                break
                            tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')
                            tiles.append(standardize_transform(tile))
                        if len(tiles) >= self.max_tiles:
                            break
                    slide.close()
                except openslide.OpenSlideError:
                    tiles = [standardize_transform(Image.open(slide_path).convert('RGB'))]
                except Exception:
                    tiles = [standardize_transform(Image.open(slide_path).convert('RGB'))]
            else:
                tiles = [standardize_transform(Image.open(slide_path).convert('RGB'))]

            if not tiles:
                raise ValueError("No tiles extracted")
            return tiles, slide_path
        except Exception as e:
            logging.error(f"Error processing slide {slide_path}: {e}")
            return [], slide_path


test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def simple_collate(batch):
    valid_batch = [item for item in batch if item[0]]
    if not valid_batch:
        return [], []
    tiles_list, paths = zip(*valid_batch)
    processed_tiles = [torch.stack([test_transform(tile) for tile in tiles]) for tiles in tiles_list]
    return processed_tiles, list(paths)


# ================================================================
# MODEL ARCHITECTURE
# ================================================================

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = Dinov2Model.from_pretrained("owkin/phikon-v2")

    def forward(self, x):
        return self.vit(pixel_values=x).last_hidden_state[:, 0]


class ClassificationHead(nn.Module):
    def __init__(self, in_dim=1024, num_classes=2, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class HierarchicalMILAggregator(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, num_layers=2,
                 num_diseases=6, num_stage_classes=0, disease_names=None):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = nn.Sequential(nn.Linear(embed_dim, 256), nn.Tanh(), nn.Linear(256, 1))
        self.disease_head = ClassificationHead(embed_dim, num_diseases)
        self.severity_heads = nn.ModuleDict()
        for name in disease_names:
            self.severity_heads[name] = ClassificationHead(embed_dim, 2)
        self.stage_head = ClassificationHead(embed_dim, num_stage_classes) if num_stage_classes > 0 else None
        self.disease_name_to_idx = {n: i for i, n in enumerate(disease_names)}
        self.idx_to_disease_name = {i: n for n, i in self.disease_name_to_idx.items()}
        self.disease_names = disease_names

    def forward(self, tile_features):
        normalized = self.pre_norm(tile_features)
        aggregated = self.transformer(normalized)
        attn_scores = self.attention(aggregated)
        attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=1)
        weighted = torch.sum(aggregated * attn_weights.unsqueeze(-1), dim=1)
        disease_logits = self.disease_head(weighted)
        severity_logits = {n: self.severity_heads[n](weighted) for n in self.disease_names}
        stage_logits = self.stage_head(weighted) if self.stage_head is not None else None
        return disease_logits, severity_logits, stage_logits, attn_weights


class Phase3Model(nn.Module):
    def __init__(self, backbone, num_diseases=6, num_stage_classes=0, disease_names=None):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.aggregator = HierarchicalMILAggregator(
            num_diseases=num_diseases,
            num_stage_classes=num_stage_classes,
            disease_names=disease_names
        )

    def forward(self, tiles, enable_gradients=False):
        all_features = []
        for batch_tiles in tiles:
            if batch_tiles.numel() == 0:
                continue
            batch_tiles = batch_tiles.to(next(self.backbone.parameters()).device)
            if enable_gradients:
                batch_features = self.backbone(batch_tiles)
            else:
                with torch.no_grad():
                    batch_features = self.backbone(batch_tiles)
            all_features.append(batch_features)
        if not all_features:
            raise ValueError("No valid tile features could be extracted.")
        all_features = torch.stack(all_features)
        return self.aggregator(all_features)


# ================================================================
# PREDICTION
# ================================================================

def predict_image(model, tiles, disease_names, stage_names):
    model.eval()
    try:
        with torch.no_grad():
            disease_logits, severity_logits, stage_logits, _ = model(tiles)
            disease_probs = F.softmax(disease_logits, dim=1)
            disease_pred_idx = torch.argmax(disease_probs, dim=1).item()
            disease_confidence = disease_probs[0, disease_pred_idx].item()
            predicted_disease = disease_names[disease_pred_idx]

            severity_probs = F.softmax(severity_logits[predicted_disease], dim=1)
            severity_pred = torch.argmax(severity_probs, dim=1).item()
            severity_confidence = severity_probs[0, severity_pred].item()
            severity_label = "Normal" if severity_pred == 0 else "Abnormal"

            stage_label = "N/A"
            stage_confidence = 0.0
            if severity_pred == 1 and stage_logits is not None:
                stage_probs = F.softmax(stage_logits, dim=1)
                stage_pred_idx = torch.argmax(stage_probs, dim=1).item()
                stage_confidence = stage_probs[0, stage_pred_idx].item()
                stage_label = stage_names.get(stage_pred_idx, f"Stage_{stage_pred_idx}")

            return {
                'disease': predicted_disease,
                'disease_confidence': disease_confidence,
                'severity': severity_label,
                'severity_confidence': severity_confidence,
                'stage': stage_label,
                'stage_confidence': stage_confidence
            }
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None


# ================================================================
# GPT EXPLANATION GENERATOR
# ================================================================

def generate_comprehensive_explanation(comprehensive_data):
    """
    Calls GPT-4o-mini to convert technical XAI metrics into a human-friendly
    explanation. Falls back to a template string if the API call fails.
    Only references Attention and GradCAM heatmaps.
    """
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        prompt = f"""You are an AI explainability assistant helping users understand how a hierarchical medical image classification model made its decision. Convert the following technical analysis into a clear, accessible explanation.

HIERARCHICAL MODEL PREDICTION:
- Region: {comprehensive_data['predicted_disease']} ({comprehensive_data['gradcam_disease_conf']:.1%} confidence)
- Status Level: {comprehensive_data['gradcam_severity']} ({comprehensive_data['gradcam_severity_conf']:.1%} confidence)
- Stage Level: {comprehensive_data['predicted_stage']} ({comprehensive_data['stage_confidence']:.1%} confidence)

GRADCAM ANALYSIS (Gradient-weighted Class Activation Mapping):
- Note: Bright/warm regions in GradCAM indicate areas that most strongly influenced the model's prediction

SPATIAL ATTENTION PATTERN AND VISUAL CHARACTERISTICS (from Attention Heatmap):
- Primary Focus: {comprehensive_data['primary_position']} (intensity: {comprehensive_data['primary_intensity']:.2f})
- Attention Hotspots: {comprehensive_data['hotspot_count']}
- Spatial Distribution: Center {comprehensive_data['center_attention']:.1f}%, Mid-region {comprehensive_data['mid_attention']:.1f}%, Periphery {comprehensive_data['periphery_attention']:.1f}%
- Clustering: {comprehensive_data['scatter_level']} scatter level with {comprehensive_data['num_clusters']} clusters
- Dominant Color: {comprehensive_data['dominant_color']} ({comprehensive_data['color_confidence']:.1f}% confidence)
- Texture Pattern: {comprehensive_data['texture_classification']}
- Texture Scores: Uniformity {comprehensive_data['uniformity']}/100, Organization {comprehensive_data['organization']}/100, Complexity {comprehensive_data['complexity']}/100, Smoothness {comprehensive_data['smoothness']}/100

CRITICAL INSTRUCTIONS:
1. Write in clear, accessible language for someone without medical or technical expertise
2. Ground ALL statements in the provided data - do NOT add medical interpretations or diagnoses
3. Explain how the two explainability methods (Attention Heatmap and GradCAM) show WHERE the model focused
4. Describe WHAT visual patterns were detected, not WHY medically
5. Keep it concise but informative (under 100 words)
6. Structure with clear sections
7. Make it conversational but professional
8. Visual Characteristics and Spatial Attention Pattern were taken from Attention Heatmap

Generate a comprehensive explanation covering: what the model decided, where it looked, what the Attention and GradCAM methods revealed, what visual characteristics were important, and how confident we can be in the decision.

Format as natural paragraphs, not bullet points."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at explaining complex AI model decisions in simple, clear language. You help users understand model behavior without making medical claims."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.warning(f"OpenAI API call failed: {e}. Using fallback template.")
        return (
            f"MODEL DECISION SUMMARY\n\n"
            f"The model classified this as '{comprehensive_data['predicted_disease']}' "
            f"with severity '{comprehensive_data['gradcam_severity']}' "
            f"and stage '{comprehensive_data['predicted_stage']}' "
            f"({comprehensive_data['stage_confidence']:.1%} confidence).\n\n"
            f"ATTENTION ANALYSIS\n\n"
            f"Primary focus: {comprehensive_data['primary_position']} region. "
            f"Attention shows {comprehensive_data['scatter_level']} scatter across "
            f"{comprehensive_data['num_clusters']} clusters. "
            f"Distribution - Center: {comprehensive_data['center_attention']:.1f}%, "
            f"Mid: {comprehensive_data['mid_attention']:.1f}%, "
            f"Periphery: {comprehensive_data['periphery_attention']:.1f}%.\n\n"
            f"GRADCAM ANALYSIS\n\n"
            f"GradCAM confidence: {comprehensive_data['gradcam_disease_conf']:.1%} (disease), "
            f"{comprehensive_data['gradcam_severity_conf']:.1%} (severity).\n\n"
            f"VISUAL PATTERNS\n\n"
            f"Dominant color: {comprehensive_data['dominant_color']}. "
            f"Texture: {comprehensive_data['texture_classification']} "
            f"(uniformity {comprehensive_data['uniformity']}/100, "
            f"smoothness {comprehensive_data['smoothness']}/100)."
        )


# ================================================================
# RENDER EXPLANATION TEXT -> RGB NUMPY ARRAY FOR ROW 3
# ================================================================

def _render_explanation_to_image(explanation_text, figsize=(16, 4)):
    """
    Renders a plain-text explanation string into an (H, W, 3) uint8 numpy
    array that fills the entire Row 3 panel in display_prediction().
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0F0F2A')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor('#0F0F2A')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.01",
        linewidth=2,
        edgecolor='#3498DB',
        facecolor='#16213E',
        transform=ax.transAxes,
        clip_on=False
    ))

    ax.text(
        0.5, 0.93,
        'Textual Explanation',
        ha='center', va='top',
        fontsize=11,
        fontweight='bold',
        color='#3498DB',
        transform=ax.transAxes
    )

    ax.add_line(plt.Line2D(
        [0.02, 0.98], [0.855, 0.855],
        transform=ax.transAxes,
        color='#3498DB',
        linewidth=1.0
    ))

    ax.text(
        0.02, 0.83,
        explanation_text,
        va='top', ha='left',
        fontsize=10,
        color='#E0E0F0',
        family='monospace',
        wrap=True,
        transform=ax.transAxes
    )

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    plt.close(fig)
    return img_array[:, :, :3]


# ================================================================
# ATTENTION OVERLAY HELPERS
# ================================================================

def _build_attention_overlay(img_array, heatmap_raw):
    """
    Takes img_array (H, W, 3) uint8 and heatmap_raw (H, W) float,
    returns an RGB overlay as np.ndarray in [0, 1].
    """
    import cv2
    img_norm = img_array.astype(np.float32) / 255.0
    hm = heatmap_raw.astype(np.float32)

    h, w = img_norm.shape[:2]
    if hm.shape != (h, w):
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)

    hm_min, hm_max = hm.min(), hm.max()
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)

    cmap = matplotlib.colormaps.get_cmap('jet')
    hm_colored = cmap(hm)[:, :, :3]
    overlay = img_norm * 0.5 + hm_colored * 0.5
    return np.clip(overlay, 0, 1)


def _preprocess_images_for_attention(image_paths):
    processed = []
    for p in image_paths:
        try:
            img_pil = Image.open(p).convert('RGB').resize((224, 224), Image.BILINEAR)
            tensor = test_transform(img_pil)
            processed.append(tensor)
        except Exception as e:
            logging.warning(f"Could not preprocess {p} for attention: {e}")
            processed.append(torch.zeros(3, 224, 224))
    return processed


# ================================================================
# DISPLAY FUNCTION  -  3-ROW LAYOUT
# Row 1 : Original Image  |  Diagnostic Report
# Row 2 : Attention Heatmap  |  GradCAM Heatmap
# Row 3 : GPT-Generated Human-Friendly Text Explanation
# ================================================================

def display_prediction(image_path, prediction,
                       heatmap_images=None,
                       heatmap_titles=None,
                       explanation_image=None):
    severity    = prediction['severity']
    accent      = '#E74C3C' if severity == 'Abnormal' else '#2ECC71'
    bg_color    = '#1A1A2E'
    panel_color = '#16213E'
    border_dim  = '#2A2A4A'

    # Exactly 2 heatmaps: Attention + GradCAM
    if heatmap_titles is None:
        heatmap_titles = ['Attention Heatmap', 'GradCAM Heatmap']

    fig = plt.figure(figsize=(16, 14), facecolor=bg_color)
    outer_gs = GridSpec(3, 1, figure=fig,
                        height_ratios=[5, 4, 3],
                        hspace=0.08)

    # ===== ROW 1: Original Image | Diagnostic Report =====
    row1_gs = GridSpecFromSubplotSpec(1, 2,
                                      subplot_spec=outer_gs[0],
                                      width_ratios=[1, 1.2],
                                      wspace=0.05)

    ax_img = fig.add_subplot(row1_gs[0])
    ax_img.set_facecolor(bg_color)
    try:
        img = Image.open(image_path).convert('RGB')
        ax_img.imshow(img)
    except Exception:
        ax_img.text(0.5, 0.5, 'WSI / Slide\n(preview unavailable)',
                    ha='center', va='center', color='white', fontsize=13,
                    transform=ax_img.transAxes)
    for spine in ax_img.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(3)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(os.path.basename(image_path), color='white',
                     fontsize=11, pad=8, fontweight='bold')

    ax_info = fig.add_subplot(row1_gs[1])
    ax_info.set_facecolor(bg_color)
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis('off')
    ax_info.text(0.5, 0.96, 'Diagnostic Report',
                 ha='center', va='top', fontsize=15, fontweight='bold',
                 color='white', transform=ax_info.transAxes)
    divider = plt.Line2D([0.05, 0.95], [0.89, 0.89],
                         transform=ax_info.transAxes,
                         color=accent, linewidth=1.5)
    ax_info.add_line(divider)

    def draw_card(ax, y, label, value, confidence, color):
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.04, y - 0.11), 0.92, 0.14,
            boxstyle="round,pad=0.01",
            linewidth=1.5, edgecolor=color,
            facecolor=panel_color, transform=ax.transAxes, clip_on=False
        ))
        ax.text(0.10, y - 0.01, label.upper(),
                ha='left', va='center', fontsize=8, color='#A0A0C0',
                fontweight='bold', transform=ax.transAxes)
        ax.text(0.10, y - 0.05, value,
                ha='left', va='center', fontsize=13, color='white',
                fontweight='bold', transform=ax.transAxes)
        if confidence > 0:
            bar_y = y - 0.09
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.08, bar_y), 0.60, 0.015,
                boxstyle="round,pad=0.001", linewidth=0,
                facecolor='#0F3460', transform=ax.transAxes, clip_on=False
            ))
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.08, bar_y), 0.60 * confidence, 0.015,
                boxstyle="round,pad=0.001", linewidth=0,
                facecolor=color, transform=ax.transAxes, clip_on=False
            ))
            ax.text(0.72, bar_y + 0.007, f'{confidence:.1%}',
                    ha='left', va='center', fontsize=9, color=color,
                    fontweight='bold', transform=ax.transAxes)

    draw_card(ax_info, 0.78,
              'Region',
              prediction['disease'].replace('_processed', '').replace('_', ' ').title(),
              prediction['disease_confidence'], '#3498DB')
    draw_card(ax_info, 0.57, 'Status', severity,
              prediction['severity_confidence'], accent)

    stage_val  = prediction['stage']
    stage_conf = prediction['stage_confidence']
    if stage_val == 'N/A':
        stage_display = 'N/A (Normal)'
        stage_conf = 0
    else:
        stage_display = (stage_val.split(' - ')[-1].replace('_', ' ').title()
                         if ' - ' in stage_val else stage_val)
    draw_card(ax_info, 0.36, 'Stage / Subtype', stage_display, stage_conf, '#F39C12')

    # ===== ROW 2: Attention Heatmap | GradCAM Heatmap (2 columns) =====
    row2_gs = GridSpecFromSubplotSpec(1, 2,
                                      subplot_spec=outer_gs[1],
                                      wspace=0.06)
    for col_idx in range(2):
        ax_hm = fig.add_subplot(row2_gs[col_idx])
        ax_hm.set_facecolor(panel_color)

        if heatmap_images and col_idx < len(heatmap_images) and heatmap_images[col_idx] is not None:
            hm = heatmap_images[col_idx]
            ax_hm.imshow(hm if isinstance(hm, np.ndarray) else np.array(hm))
        else:
            ax_hm.set_xlim(0, 1)
            ax_hm.set_ylim(0, 1)
            ax_hm.add_patch(mpatches.FancyBboxPatch(
                (0.05, 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.02",
                linewidth=1.5, linestyle='--',
                edgecolor='#4A4A6A', facecolor='#0F0F2A',
                transform=ax_hm.transAxes, clip_on=False
            ))
            ax_hm.text(0.5, 0.5, '[ Heatmap\nPlaceholder ]',
                       ha='center', va='center',
                       color='#4A4A6A', fontsize=9, fontstyle='italic',
                       transform=ax_hm.transAxes)

        title = heatmap_titles[col_idx] if col_idx < len(heatmap_titles) else f'Heatmap {col_idx+1}'
        ax_hm.set_title(title, color='#A0A0C0', fontsize=9, fontweight='bold', pad=5)
        for spine in ax_hm.spines.values():
            spine.set_edgecolor(border_dim)
            spine.set_linewidth(1.2)
        ax_hm.set_xticks([])
        ax_hm.set_yticks([])

    # ===== ROW 3: GPT Explanation =====
    ax_text = fig.add_subplot(outer_gs[2])
    ax_text.set_facecolor(panel_color)

    if explanation_image is not None:
        exp_img = explanation_image if isinstance(explanation_image, np.ndarray) \
                  else np.array(explanation_image)
        ax_text.imshow(exp_img, aspect='auto')
        ax_text.set_xticks([])
        ax_text.set_yticks([])
    else:
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        ax_text.axis('off')
        ax_text.add_patch(mpatches.FancyBboxPatch(
            (0.01, 0.05), 0.98, 0.90,
            boxstyle="round,pad=0.02",
            linewidth=1.5, linestyle='--',
            edgecolor='#4A4A6A', facecolor='#0F0F2A',
            transform=ax_text.transAxes, clip_on=False
        ))
        ax_text.text(0.5, 0.80,
                     'Human-Friendly Text Explanation',
                     ha='center', va='center',
                     color='#A0A0C0', fontsize=11, fontweight='bold',
                     transform=ax_text.transAxes)
        ax_text.text(0.5, 0.42,
                     '[ Textual Explanation Placeholder ]\n\n'
                     'The model focused on [extracted features]\n'
                     'because [rule-based reasoning] ...',
                     ha='center', va='center',
                     color='#3A3A5A', fontsize=10, fontstyle='italic',
                     transform=ax_text.transAxes)

    for spine in ax_text.spines.values():
        spine.set_edgecolor(border_dim)
        spine.set_linewidth(1.2)

    plt.tight_layout(pad=1.2)
    plt.show()
    print()


# ================================================================
# MODEL LOADER
# ================================================================

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict']

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    severity_head_names = set()
    for k in state_dict.keys():
        if k.startswith('aggregator.severity_heads.'):
            parts = k.split('.')
            if len(parts) > 2:
                severity_head_names.add(parts[2])
    disease_names = sorted(list(severity_head_names))

    num_diseases = state_dict['aggregator.disease_head.classifier.3.weight'].shape[0]
    num_stage_classes = (
        state_dict['aggregator.stage_head.classifier.3.weight'].shape[0]
        if 'aggregator.stage_head.classifier.3.weight' in state_dict else 0
    )

    print(f"\nModel config -> diseases: {num_diseases} | stages: {num_stage_classes}")
    print(f"Classes: {', '.join(disease_names)}\n")

    backbone = ViTBackbone()
    model = Phase3Model(backbone, num_diseases=num_diseases,
                        num_stage_classes=num_stage_classes,
                        disease_names=disease_names).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully!\n")
    return model, disease_names


# ================================================================
# MAIN INFERENCE PIPELINE
# ================================================================

def run_inference():
    print("\n" + "=" * 70)
    print("   HIERARCHICAL MIL MODEL  -  PATHOLOGY INFERENCE")
    print("=" * 70)
    print("\nOptions:")
    print("  1.  Single image  (provide full path to one image file)")
    print("  2.  Folder        (provide path to a folder; all images processed)")

    choice = input("\nSelect option (1 / 2): ").strip()

    if choice == '1':
        image_path = input("Enter image path: ").strip()
        if not os.path.isfile(image_path):
            print(f"ERROR: File not found -> {image_path}")
            return
        all_images = [image_path]
    elif choice == '2':
        folder_path = input("Enter folder path: ").strip()
        if not os.path.isdir(folder_path):
            print(f"ERROR: Folder not found -> {folder_path}")
            return
        all_images = collect_images_from_folder(folder_path)
        if not all_images:
            print("No valid images found in the folder.")
            return
        print(f"Found {len(all_images)} image(s).")
    else:
        print("Invalid option.")
        return

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------
    print("\nLoading model ...")
    model, disease_names = load_model()
    if model is None:
        return

    # ----------------------------------------------------------
    # STEP 1: Run Attention and GradCAM analyses only
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Running Attention Analysis ...")
    print("=" * 70)
    processed_images = _preprocess_images_for_attention(all_images)
    attn_results = run_attention_analysis(
        attention_extractor,
        device,
        all_images,
        processed_images,
        DISEASE_CLASS_MAPPING,
        STAGE_CLASS_MAPPING
    )

    print("\n" + "=" * 70)
    print("Running GradCAM Analysis ...")
    print("=" * 70)
    gradcam_results = run_tri_head_gradcam_plus_plus_analysis(model, device, all_images)

    # ----------------------------------------------------------
    # STEP 2: Feature extraction from attention heatmaps
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Running Feature Extraction ...")
    print("=" * 70)

    explanations_list = []

    for i, attention_result in enumerate(attn_results):
        print(f"  [{i+1}/{len(attn_results)}] Extracting features: {attention_result['filename']}")

        extractor = HeatmapFeatureExtractor(attention_result)

        bright    = extractor.get_brightest_region()
        scatter   = extractor.get_activation_scatter()
        dom_color = extractor.get_dominant_focus_color()
        texture   = extractor.get_texture_analysis()

        explanations_list.append({
            "brightest":      bright,
            "scatter":        scatter,
            "dominant_color": dom_color,
            "texture":        texture,
        })

        print(f"  ✅ Position={bright['primary_hotspot']['position']}, "
              f"Scatter={scatter['scatter_level']}, "
              f"Color={dom_color['dominant_color_name']}, "
              f"Texture={texture['texture_classification']}")

    # ----------------------------------------------------------
    # STEP 3: Standard inference loop + GPT explanation + display
    # ----------------------------------------------------------
    dataset    = SimpleSlideDataset(all_images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=simple_collate, num_workers=2, pin_memory=True)

    results = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        tiles, paths = batch
        if not tiles or not paths:
            continue

        slide_path = paths[0]

        try:
            img_idx = all_images.index(slide_path)
        except ValueError:
            img_idx = batch_idx

        prediction = predict_image(model, tiles, disease_names, STAGE_NAMES)
        if prediction is None:
            print(f"Failed to process: {slide_path}")
            continue

        # ---- Build the 2 overlay images: Attention + GradCAM ----
        attn_overlay = None
        if attn_results and img_idx < len(attn_results):
            ar = attn_results[img_idx]
            attn_overlay = _build_attention_overlay(ar['image'], ar['attention_heatmap'])

        gradcam_overlay = None
        if gradcam_results and img_idx < len(gradcam_results):
            gradcam_overlay = gradcam_results[img_idx]['union']['heatmap_overlay']

        # ---- Build comprehensive_data dict for GPT ----
        exp         = explanations_list[img_idx]
        bright      = exp['brightest']
        scatter_res = exp['scatter']
        dom         = exp['dominant_color']
        texture     = exp['texture']

        # GradCAM-sourced confidence and severity values
        gradcam_disease_conf  = gradcam_results[img_idx]['level1_disease']['confidence'] \
                                if gradcam_results else 0.0
        gradcam_severity_conf = gradcam_results[img_idx]['level2_severity']['confidence'] \
                                if gradcam_results else 0.0
        gradcam_severity      = gradcam_results[img_idx]['level2_severity']['predicted_class'] \
                                if gradcam_results else 'N/A'

        comprehensive_data = {
            'predicted_disease':      prediction['disease'],
            'gradcam_severity':       gradcam_severity,
            'predicted_stage':        prediction['stage'],
            'stage_confidence':       prediction['stage_confidence'],
            'gradcam_disease_conf':   gradcam_disease_conf,
            'gradcam_severity_conf':  gradcam_severity_conf,
            'primary_position':       bright['primary_hotspot']['position'],
            'primary_intensity':      bright['primary_hotspot']['intensity'],
            'hotspot_count':          bright['hotspot_count'],
            'center_attention':       bright['spatial_coverage']['center_attention'],
            'mid_attention':          bright['spatial_coverage']['mid_region_attention'],
            'periphery_attention':    bright['spatial_coverage']['periphery_attention'],
            'scatter_level':          scatter_res['scatter_level'],
            'num_clusters':           scatter_res['num_clusters'],
            'dominant_color':         dom['dominant_color_name'],
            'color_confidence':       dom['color_confidence'],
            'texture_classification': texture['texture_classification'],
            'uniformity':             texture['texture_scores']['uniformity'],
            'organization':           texture['texture_scores']['organization'],
            'complexity':             texture['texture_scores']['complexity'],
            'smoothness':             texture['texture_scores']['smoothness'],
        }

        # ---- Generate GPT explanation -> render to image for Row 3 ----
        print(f"\n  Generating GPT explanation for image {img_idx + 1} ...")
        explanation_text  = generate_comprehensive_explanation(comprehensive_data)
        explanation_image = _render_explanation_to_image(explanation_text)
        time.sleep(0.5)

        # ---- Display: Original | Report | Attention | GradCAM | Explanation ----
        display_prediction(
            image_path        = slide_path,
            prediction        = prediction,
            heatmap_images    = [attn_overlay, gradcam_overlay],
            heatmap_titles    = ['Attention Heatmap', 'GradCAM Heatmap'],
            explanation_image = explanation_image
        )

        results.append({
            'image_path':          slide_path,
            'image_name':          os.path.basename(slide_path),
            'predicted_disease':   prediction['disease'],
            'disease_confidence':  prediction['disease_confidence'],
            'predicted_severity':  prediction['severity'],
            'severity_confidence': prediction['severity_confidence'],
            'predicted_stage':     prediction['stage'],
            'stage_confidence':    prediction['stage_confidence']
        })

    if results:
        df       = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, "inference_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\nResults saved -> {out_path}")
        print(f"Total processed: {len(results)}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    run_inference()