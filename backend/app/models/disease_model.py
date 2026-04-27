"""
Disease classification model wrapper.
Loads the trained hierarchical MIL model and provides inference interface.
Production-ready: Uses lazy loading - models only load on first inference, never during build.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ViTImageProcessor, AutoConfig
from huggingface_hub import hf_hub_download
import logging
import os

logger = logging.getLogger(__name__)

# Global model instance (loaded once on startup)
_model_instance = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HuggingFace-only model sources
HF_BACKBONE_REPO = "owkin/phikon-v2"
HF_CHECKPOINT_REPO = "kimaan28/Diagnostic_model"
HF_CHECKPOINT_FILENAME = "phase3_mil_best.pth"

# CRITICAL: Disease names in EXACT TRAINING ORDER
# Must match the order used during model training
# Extracted from Kaggle inference code to ensure consistency
TRAINING_DISEASE_NAMES = [
    'Breast_cancer',
    'annrbc-anemia_processed',
    'colon_processed',
    'leukemia_processed',
    'lung_processed',
    'oral-cancer_processed',
    'ovarian-cancer_processed',
    'sickle-cell-new_processed',
    'thalassemia_processed'
]


class ViTBackbone(nn.Module):
    """DinoV2 Vision Transformer backbone - downloads from HuggingFace on first use."""
    
    def __init__(self):
        super().__init__()
        logger.info(f"Initializing ViT backbone...")
        self.vit = None
        self._loaded = False
    
    def _load_backbone(self):
        """Lazy load the backbone model - checks local directory first, then HuggingFace."""
        if self._loaded:
            return
        
        try:
            # Check for local model directory first (needs config.json + model.safetensors)
            local_model_dir = os.path.join(
                os.path.dirname(__file__), 
                "../phikonv2"
            )
            
            has_config = os.path.exists(os.path.join(local_model_dir, "config.json"))
            has_model = os.path.exists(os.path.join(local_model_dir, "model.safetensors"))
            
            if has_config and has_model:
                logger.info(f"📂 Found complete local model at: {local_model_dir}")
                logger.info(f"📥 Loading ViT backbone from local directory...")
                try:
                    self.vit = AutoModel.from_pretrained(
                        local_model_dir,
                        trust_remote_code=True,
                        local_files_only=True  # Don't download
                    )
                    logger.info(f"✅ ViT backbone loaded from LOCAL: {local_model_dir}")
                    self._loaded = True
                    return
                except Exception as local_e:
                    logger.warning(f"⚠️  Failed to load from local: {local_e}")
            else:
                if not has_config:
                    logger.info(f"⚠️  Missing config.json in {local_model_dir}")
                if not has_model:
                    logger.info(f"⚠️  Missing model.safetensors in {local_model_dir}")
            
            # Fallback to HuggingFace
            logger.info(f"☁️  Downloading ViT backbone from HuggingFace: {HF_BACKBONE_REPO}")
            
            self.vit = AutoModel.from_pretrained(
                HF_BACKBONE_REPO,
                trust_remote_code=True,
                local_files_only=False,  # Allow downloading
                force_download=False  # Use cache if available
            )
            logger.info(f"✅ ViT backbone loaded from HuggingFace: {HF_BACKBONE_REPO}")
            self._loaded = True
        except Exception as e:
            logger.error(f"❌ Failed to load ViT model: {e}")
            raise RuntimeError(
                f"Cannot load ViT backbone. "
                f"Tried: (1) Local directory at app/phikonv2/ (needs config.json + model.safetensors), "
                f"(2) HuggingFace {HF_BACKBONE_REPO}. "
                f"Error: {e}. "
                f"To fix: Ensure internet access for HuggingFace download."
            )
    
    def forward(self, x):
        """Extract features from image tiles."""
        if not self._loaded:
            self._load_backbone()
        return self.vit(pixel_values=x).last_hidden_state[:, 0]


class ClassificationHead(nn.Module):
    """Multi-layer classification head."""
    
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
    """
    Multiple Instance Learning (MIL) aggregator with hierarchical classification heads.
    
    Three-level hierarchy:
    1. Disease: What disease is present? (multi-class)
    2. Severity: How severe is it? Normal (0) vs Abnormal (1) per disease
    3. Stage: If abnormal, what stage is it? (multi-class per disease)
    """
    
    def __init__(self, embed_dim=1024, num_heads=8, num_layers=2, 
                 num_diseases=6, num_stage_classes=0, disease_names=None):
        super().__init__()
        
        # Layer normalization before transformer
        self.pre_norm = nn.LayerNorm(embed_dim)
        
        # Transformer for multiple instance learning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention mechanism to weight tile importance
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Level 1: Disease classification (main head)
        self.disease_head = ClassificationHead(embed_dim, num_diseases)
        
        # Level 2: Severity heads (one per disease)
        self.severity_heads = nn.ModuleDict()
        for disease_name in disease_names:
            self.severity_heads[disease_name] = ClassificationHead(embed_dim, 2)  # 0=normal, 1=abnormal
        
        # Level 3: Stage head (if applicable)
        self.stage_head = ClassificationHead(embed_dim, num_stage_classes) if num_stage_classes > 0 else None
        
        # Store mappings
        self.disease_name_to_idx = {name: idx for idx, name in enumerate(disease_names)}
        self.idx_to_disease_name = {idx: name for name, idx in self.disease_name_to_idx.items()}
        self.disease_names = disease_names
    
    def forward(self, tile_features):
        """
        Forward pass through aggregator.
        
        Args:
            tile_features: Tensor of shape (batch_size, num_tiles, embed_dim)
        
        Returns:
            disease_logits: (batch_size, num_diseases)
            severity_logits: dict of disease_name -> (batch_size, 2)
            stage_logits: (batch_size, num_stage_classes) if stage head exists
            attention_weights: (batch_size, num_tiles) - importance of each tile
        """
        # Normalize features
        normalized_features = self.pre_norm(tile_features)
        
        # Aggregate tiles using transformer
        aggregated = self.transformer(normalized_features)
        
        # Compute attention weights for each tile
        attention_scores = self.attention(aggregated)  # (batch_size, num_tiles, 1)
        attention_weights = torch.softmax(attention_scores.squeeze(-1), dim=1)  # (batch_size, num_tiles)
        
        # Weighted aggregation of tile features
        weighted_features = torch.sum(
            aggregated * attention_weights.unsqueeze(-1), 
            dim=1
        )  # (batch_size, embed_dim)
        
        # Level 1: Disease prediction
        disease_logits = self.disease_head(weighted_features)  # (batch_size, num_diseases)
        
        # Level 2: Severity predictions (per disease)
        severity_logits = {}
        for disease_name in self.disease_names:
            severity_logits[disease_name] = self.severity_heads[disease_name](weighted_features)
        
        # Level 3: Stage prediction (if applicable)
        stage_logits = self.stage_head(weighted_features) if self.stage_head is not None else None
        
        return disease_logits, severity_logits, stage_logits, attention_weights


class Phase3Model(nn.Module):
    """
    Complete hierarchical disease classification model.
    
    Components:
    - Backbone: DinoV2 ViT for feature extraction
    - Aggregator: Hierarchical MIL aggregator
    """
    
    def __init__(self, backbone=None, num_diseases=6, num_stage_classes=0, disease_names=None):
        super().__init__()
        
        # Create backbone if not provided
        if backbone is None:
            backbone = ViTBackbone()
        
        # Frozen backbone
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Trainable aggregator
        self.aggregator = HierarchicalMILAggregator(
            num_diseases=num_diseases,
            num_stage_classes=num_stage_classes,
            disease_names=disease_names
        )
    
    def forward(self, tiles, enable_gradients=False):
        """
        Forward pass through model.
        
        Args:
            tiles: List of tensors, each of shape (num_tiles, 3, 224, 224)
            enable_gradients: Whether to enable gradients for the backbone (for GradCAM)
        
        Returns:
            disease_logits: (batch_size, num_diseases)
            severity_logits: dict of disease_name -> (batch_size, 2)
            stage_logits: (batch_size, num_stage_classes) if stage head exists
            attention_weights: (batch_size, num_tiles) - importance of each tile
        """
        all_features = []
        
        # Extract features for each slide (batch of tiles)
        for batch_tiles in tiles:
            if batch_tiles.numel() == 0:
                continue
            
            # Move to device (use aggregator device since backbone is frozen)
            aggregator_device = next(self.aggregator.parameters()).device
            batch_tiles = batch_tiles.to(aggregator_device)
            
            # Extract features using frozen backbone
            if enable_gradients:
                batch_features = self.backbone(batch_tiles)  # (num_tiles, embed_dim)
            else:
                with torch.no_grad():
                    batch_features = self.backbone(batch_tiles)  # (num_tiles, embed_dim)
            
            all_features.append(batch_features)
        
        if not all_features:
            raise ValueError("No valid tile features could be extracted.")
        
        # Stack features (batch_size, num_tiles, embed_dim)
        all_features = torch.stack(all_features)
        
        # Forward through aggregator
        disease_logits, severity_logits, stage_logits, attn_weights = self.aggregator(all_features)
        
        return disease_logits, severity_logits, stage_logits, attn_weights


class DiseaseModelWrapper:
    """
    High-level wrapper for disease classification model.
    Handles loading, inference, and result processing.
    """
    
    def __init__(self, model_path: str, device=None):
        """
        Initialize model wrapper.
        
        Args:
            model_path: Path to checkpoint file (phase3_mil_best.pth)
            device: Torch device (cuda or cpu)
        """
        self.device = device or _device
        self.model_path = model_path
        self.model = None
        self.disease_names = None
        self.num_diseases = None
        self.num_stage_classes = None
        
        logger.info(f"Initializing disease model on device: {self.device}")
    
    def load(self):
        """Load model checkpoint - checks local first, then HuggingFace."""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            # Check for local checkpoint files first
            local_checkpoint_paths = [
                os.path.join(os.path.dirname(__file__), "../ml_models/phase3_mil_best.pth"),
                os.path.join(os.path.dirname(__file__), "../phikonv2/phase3_mil_best.pth"),
                "./app/ml_models/phase3_mil_best.pth",
                "./ml_models/phase3_mil_best.pth",
            ]
            
            checkpoint_file = None
            for path in local_checkpoint_paths:
                if os.path.exists(path):
                    checkpoint_file = os.path.abspath(path)
                    logger.info(f"📂 Found local checkpoint at: {checkpoint_file}")
                    logger.info(f"📥 Loading model from LOCAL file...")
                    break
            
            # If no local checkpoint, try downloading from HuggingFace
            if not checkpoint_file:
                logger.info(f"⚠️  Local checkpoint not found in:")
                for path in local_checkpoint_paths:
                    logger.info(f"    - {path}")
                logger.info(f"☁️  Attempting to download checkpoint from HuggingFace: {self.model_path}")
                token = os.getenv("HF_TOKEN")
                
                if token:
                    logger.info(f"✅ HF_TOKEN is set (length: {len(token)} chars)")
                else:
                    logger.warning(f"⚠️  HF_TOKEN is NOT set - will attempt anonymous access")
                
                logger.info(f"📥 Calling hf_hub_download(repo_id='{self.model_path}', filename='{HF_CHECKPOINT_FILENAME}')")

                try:
                    # Try standard hf_hub_download first
                    checkpoint_file = hf_hub_download(
                        repo_id=self.model_path,
                        filename=HF_CHECKPOINT_FILENAME,
                        token=token,
                        revision="main",
                        resume_download=True,
                        force_download=False
                    )
                except Exception as e1:
                    logger.warning(f"⚠️  Standard download failed: {e1}")
                    logger.info("🔄 Attempting alternative download with force_download=True...")
                    try:
                        checkpoint_file = hf_hub_download(
                            repo_id=self.model_path,
                            filename=HF_CHECKPOINT_FILENAME,
                            token=token,
                            revision="main",
                            force_download=True,
                            resume_download=True
                        )
                    except Exception as e2:
                        logger.error(f"❌ Both download methods failed!")
                        logger.error(f"  Standard: {e1}")
                        logger.error(f"  Force: {e2}")
                        raise RuntimeError(
                            f"Failed to load model. Tried: (1) Local files, (2) HuggingFace {self.model_path}. "
                            f"Error: {e2}. "
                            f"To fix: Place phase3_mil_best.pth in app/ml_models/ OR set HF_TOKEN for HuggingFace."
                        ) from e2
            
            logger.info(f"✅ Checkpoint file ready: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            
            state_dict = checkpoint['model_state_dict']
            
            # Handle DataParallel wrapper (remove 'module.' prefix)
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {
                    k.replace('module.', ''): v 
                    for k, v in state_dict.items()
                }
            
            # CRITICAL FIX: Use training order, NOT sorted order
            # Validate that checkpoint has exactly these disease names
            severity_head_names = set()
            for k in state_dict.keys():
                if k.startswith('aggregator.severity_heads.'):
                    parts = k.split('.')
                    if len(parts) > 2:
                        severity_head_names.add(parts[2])
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 DISEASE NAME VALIDATION:")
            logger.info(f"   Expected (training order): {TRAINING_DISEASE_NAMES}")
            logger.info(f"   Found in checkpoint: {sorted(severity_head_names)}")
            logger.info(f"{'='*70}\n")
            
            # Verify all expected disease names are in checkpoint
            for disease_name in TRAINING_DISEASE_NAMES:
                if disease_name not in severity_head_names:
                    raise ValueError(
                        f"Disease '{disease_name}' from training order not found in checkpoint! "
                        f"Found: {severity_head_names}"
                    )
            
            # Use TRAINING order, NOT sorted order
            self.disease_names = TRAINING_DISEASE_NAMES
            logger.info(f"✅ Using disease names in TRAINING order: {self.disease_names}")
            
            # Get number of diseases from disease head output shape
            self.num_diseases = state_dict['aggregator.disease_head.classifier.3.weight'].shape[0]
            logger.info(f"   Number of diseases in checkpoint: {self.num_diseases}")
            
            # Verify consistency
            if len(self.disease_names) != self.num_diseases:
                raise ValueError(
                    f"Mismatch: {len(self.disease_names)} disease names but "
                    f"{self.num_diseases} output classes in checkpoint"
                )
            
            # Get number of stage classes
            if 'aggregator.stage_head.classifier.3.weight' in state_dict:
                self.num_stage_classes = state_dict['aggregator.stage_head.classifier.3.weight'].shape[0]
            else:
                self.num_stage_classes = 0
            logger.info(f"   Number of stage classes: {self.num_stage_classes}\n")
            
            # Create model
            self.model = Phase3Model(
                backbone=None,  # Will create ViTBackbone with lazy loading
                num_diseases=self.num_diseases,
                num_stage_classes=self.num_stage_classes,
                disease_names=self.disease_names
            ).to(self.device)
            
            # CRITICAL FIX: The Kaggle code loaded the checkpoint with strict=True, 
            # meaning the checkpoint contains backbone weights (which may be fine-tuned).
            # We must force the backbone to initialize its AutoModel before load_state_dict
            # so the backbone weights in the checkpoint are actually loaded instead of ignored.
            logger.info("📥 Initializing ViT backbone before loading checkpoint...")
            self.model.backbone._load_backbone()
            self.model.backbone = self.model.backbone.to(self.device)
            
            # Load weights
            logger.info("📥 Loading model state_dict...")
            result = self.model.load_state_dict(state_dict, strict=False)
            
            if result.missing_keys:
                logger.warning(f"⚠️  Missing keys (will use default): {len(result.missing_keys)} keys")
            if result.unexpected_keys:
                logger.warning(f"⚠️  Unexpected keys (skipped from checkpoint): {len(result.unexpected_keys)} keys")
                logger.info("    This is OK - checkpoint has frozen backbone weights that we don't load")
            
            self.model.eval()
            logger.info("✅ Model loaded successfully\n")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}", exc_info=True)
            raise
    
    def predict(self, tiles: torch.Tensor) -> dict:
        """
        Run inference on image tiles.
        
        Args:
            tiles: List of tensors or single tensor
                   Shape: (num_tiles, 3, 224, 224) or list of such tensors
        
        Returns:
            dict with:
                - disease_idx: Predicted disease index
                - disease_name: Predicted disease name (from training order)
                - disease_confidence: Confidence score [0, 1]
                - severity_idx: 0=normal, 1=abnormal
                - severity_name: "Normal" or "Abnormal"
                - severity_confidence: Confidence score [0, 1]
                - stage_idx: Stage index (if abnormal), else None
                - stage_confidence: Confidence score (if abnormal), else None
                - attention_weights: Importance of each tile
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not isinstance(tiles, list):
            tiles = [tiles]
        
        with torch.no_grad():
            try:
                disease_logits, severity_logits, stage_logits, attn_weights = self.model(tiles)
                
                # Get predictions
                batch_size = disease_logits.shape[0]
                results = []
                
                for i in range(batch_size):
                    # Level 1: Disease
                    # CRITICAL: disease_idx is the index in our disease_names list (training order)
                    disease_probs = F.softmax(disease_logits[i:i+1], dim=1)
                    disease_idx = torch.argmax(disease_probs, dim=1).item()
                    disease_confidence = disease_probs[0, disease_idx].item()
                    disease_name = self.disease_names[disease_idx]  # Correctly maps to training order
                    
                    logger.debug(f"Disease prediction: idx={disease_idx}, name={disease_name}, conf={disease_confidence:.4f}")
                    
                    # Level 2: Severity
                    # Use disease_name to get correct severity head
                    severity_logits_for_disease = severity_logits[disease_name][i:i+1]
                    severity_probs = F.softmax(severity_logits_for_disease, dim=1)
                    severity_idx = torch.argmax(severity_probs, dim=1).item()
                    severity_confidence = severity_probs[0, severity_idx].item()
                    severity_name = "Normal" if severity_idx == 0 else "Abnormal"
                    
                    logger.debug(f"Severity prediction: idx={severity_idx}, name={severity_name}, conf={severity_confidence:.4f}")
                    
                    # Level 3: Stage (only if abnormal)
                    stage_idx = None
                    stage_name = None
                    stage_confidence = None
                    
                    if severity_idx == 1 and stage_logits is not None:  # Abnormal
                        stage_probs = F.softmax(stage_logits[i:i+1], dim=1)
                        stage_idx = torch.argmax(stage_probs, dim=1).item()
                        stage_confidence = stage_probs[0, stage_idx].item()
                        stage_name = f"Stage {stage_idx}"
                        
                        logger.debug(f"Stage prediction: idx={stage_idx}, conf={stage_confidence:.4f}")
                    
                    result = {
                        'disease_idx': disease_idx,
                        'disease_name': disease_name,
                        'disease_confidence': disease_confidence,
                        'severity_idx': severity_idx,
                        'severity_name': severity_name,
                        'severity_confidence': severity_confidence,
                        'stage_idx': stage_idx,
                        'stage_name': stage_name,
                        'stage_confidence': stage_confidence,
                        'attention_weights': attn_weights[i].cpu().numpy().tolist() if attn_weights is not None else []
                    }
                    results.append(result)
                
                # Return first result for single input
                return results[0] if len(results) == 1 else results
                
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                raise


def load_model(model_path_or_hf_repo: str = None) -> DiseaseModelWrapper:
    """
    Load the disease classification model.
    
    Args:
        model_path_or_hf_repo: HuggingFace repo ID containing phase3_mil_best.pth.
            Defaults to "kimaan28/Diagnostic_model".
    
    Returns:
        Loaded model wrapper instance
    """
    global _model_instance
    
    if _model_instance is not None:
        logger.info("Model already loaded, returning cached instance")
        return _model_instance
    
    chosen_source = model_path_or_hf_repo or HF_CHECKPOINT_REPO
    logger.info(f"Using HuggingFace checkpoint repo: {chosen_source}")
    
    wrapper = DiseaseModelWrapper(chosen_source, device=_device)
    try:
        wrapper.load()
    except Exception as e:
        token_status = "SET (length: %d)" % len(os.getenv("HF_TOKEN", "")) if os.getenv("HF_TOKEN") else "NOT SET"
        raise RuntimeError(
            "Model initialization failed from HuggingFace. "
            f"Checkpoint repo: {chosen_source} | "
            f"Backbone repo: {HF_BACKBONE_REPO} | "
            f"HF_TOKEN: {token_status} | "
            "If 500 error persists, check: (1) HF_TOKEN env var is set, "
            "(2) Railway has outbound internet to huggingface.co, "
            "(3) Repos exist and are accessible. "
            f"Original error: {e}"
        ) from e

    _model_instance = wrapper
    return _model_instance


def get_model() -> DiseaseModelWrapper:
    """Get model instance, lazily loading it if needed."""
    global _model_instance
    if _model_instance is None:
        logger.info("Model cache empty, lazily loading model")
        return load_model()
    return _model_instance