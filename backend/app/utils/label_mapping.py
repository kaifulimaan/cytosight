"""
Label mappings for disease classification model predictions.
Maps numeric predictions to readable disease names and stages.

CRITICAL: Disease order must match training order, not alphabetical!
"""

import logging
from typing import Dict, Tuple, Optional
from app.models.disease_model import TRAINING_DISEASE_NAMES

logger = logging.getLogger(__name__)

# Kaggle dataset disease labels
DEFAULT_DISEASE_LABELS = {
    "Breast_cancer": {
        "name": "Breast Cancer",
        "description": "Breast cancer detection and classification"
    },
    "annrbc-anemia_processed": {
        "name": "Anemia",
        "description": "Abnormal nucleated RBC anemia detection"
    },
    "colon_processed": {
        "name": "Colon Cancer",
        "description": "Colorectal cancer detection"
    },
    "leukemia_processed": {
        "name": "Leukemia",
        "description": "Leukemia detection and classification"
    },
    "lung_processed": {
        "name": "Lung Cancer",
        "description": "Lung cancer detection and classification"
    },
    "oral-cancer_processed": {
        "name": "Oral Cancer",
        "description": "Oral cancer detection"
    },
    "ovarian-cancer_processed": {
        "name": "Ovarian Cancer",
        "description": "Ovarian cancer detection and classification"
    },
    "sickle-cell-new_processed": {
        "name": "Sickle Cell Disease",
        "description": "Sickle cell disease detection"
    },
    "thalassemia_processed": {
        "name": "Thalassemia",
        "description": "Thalassemia detection"
    }
}

# Severity labels (same for all diseases)
SEVERITY_LABELS = {
    0: {
        "name": "Normal",
        "description": "No abnormalities detected",
        "severity_level": 0
    },
    1: {
        "name": "Abnormal",
        "description": "Abnormalities detected",
        "severity_level": 1
    }
}

# Stage labels per disease (only used when severity=abnormal)
# Based on Kaggle dataset stages
STAGE_LABELS = {
    "Breast_cancer": {
        "names": {
            0: "Ductal Carcinoma",
            1: "Lobular Carcinoma",
            2: "Mucinous Carcinoma",
            3: "Papillary Carcinoma"
        },
        "description": "Histological subtypes of breast cancer"
    },
    "annrbc-anemia_processed": {
        "names": {},
        "description": "No stage classification"
    },
    "colon_processed": {
        "names": {},
        "description": "No stage classification"
    },
    "leukemia_processed": {
        "names": {
            0: "Early Stage",
            1: "Pre-Stage",
            2: "Pro-Stage"
        },
        "description": "Leukemia developmental stages"
    },
    "lung_processed": {
        "names": {
            0: "Adenocarcinoma",
            1: "Squamous Cell Carcinoma"
        },
        "description": "Histological subtypes of lung cancer"
    },
    "oral-cancer_processed": {
        "names": {},
        "description": "No stage classification"
    },
    "ovarian-cancer_processed": {
        "names": {
            0: "Clear Cell (CC)",
            1: "Endometrioid (EC)",
            2: "High-Grade Serous (HGSC)",
            3: "Low-Grade Serous (LGSC)",
            4: "Mucinous (MC)"
        },
        "description": "Histological subtypes of ovarian cancer"
    },
    "sickle-cell-new_processed": {
        "names": {},
        "description": "No stage classification"
    },
    "thalassemia_processed": {
        "names": {},
        "description": "No stage classification"
    }
}


GLOBAL_STAGE_LABELS = {
    0: {"disease_key": "Breast_cancer", "name": "Ductal Carcinoma"},
    1: {"disease_key": "Breast_cancer", "name": "Lobular Carcinoma"},
    2: {"disease_key": "Breast_cancer", "name": "Mucinous Carcinoma"},
    3: {"disease_key": "Breast_cancer", "name": "Papillary Carcinoma"},
    4: {"disease_key": "leukemia_processed", "name": "Early Stage"},
    5: {"disease_key": "leukemia_processed", "name": "Pre-Stage"},
    6: {"disease_key": "leukemia_processed", "name": "Pro-Stage"},
    7: {"disease_key": "lung_processed", "name": "Adenocarcinoma"},
    8: {"disease_key": "lung_processed", "name": "Squamous Cell Carcinoma"},
    9: {"disease_key": "ovarian-cancer_processed", "name": "Clear Cell (CC)"},
    10: {"disease_key": "ovarian-cancer_processed", "name": "Endometrioid (EC)"},
    11: {"disease_key": "ovarian-cancer_processed", "name": "High-Grade Serous (HGSC)"},
    12: {"disease_key": "ovarian-cancer_processed", "name": "Low-Grade Serous (LGSC)"},
    13: {"disease_key": "ovarian-cancer_processed", "name": "Mucinous (MC)"},
}


class LabelMapper:
    """Maps model predictions to human-readable labels."""
    
    def __init__(self, 
                 disease_names: list = None,
                 disease_labels: dict = None,
                 stage_labels: dict = None):
        
        # Use training order if provided, otherwise use default
        self.disease_names = disease_names or TRAINING_DISEASE_NAMES
        self.disease_labels = disease_labels or DEFAULT_DISEASE_LABELS
        self.stage_labels = stage_labels or STAGE_LABELS
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 LABEL MAPPER INITIALIZATION:")
        logger.info(f"   Disease names (in order): {self.disease_names}")
        logger.info(f"   Number of diseases: {len(self.disease_names)}")
        logger.info(f"{'='*70}\n")
        
        # Create index mappings from disease names
        # Index = position in disease_names list (training order)
        self.disease_name_to_idx = {name: idx for idx, name in enumerate(self.disease_names)}
        self.idx_to_disease_name = {idx: name for name, idx in self.disease_name_to_idx.items()}
        
        # Debug: Show index mapping
        logger.debug(f"Index to disease name mapping:")
        for idx, name in self.idx_to_disease_name.items():
            logger.debug(f"  {idx} -> {name}")
    
    def get_disease_name(self, disease_idx: int) -> str:
        """
        Get display name for disease index.
        
        Args:
            disease_idx: Disease prediction index (position in training order)
        
        Returns:
            Readable disease name
        """
        if disease_idx not in self.idx_to_disease_name:
            logger.warning(f"⚠️  Unknown disease index: {disease_idx}")
            return f"Unknown Disease ({disease_idx})"
        
        disease_key = self.idx_to_disease_name[disease_idx]
        
        if disease_key in self.disease_labels:
            return self.disease_labels[disease_key].get("name", disease_key)
        
        return disease_key
    
    def get_disease_info(self, disease_idx: int) -> dict:
       
        if disease_idx not in self.idx_to_disease_name:
            logger.warning(f"⚠️  Unknown disease index: {disease_idx}")
            return {
                "key": "unknown",
                "name": f"Unknown Disease",
                "description": f"Disease index {disease_idx} not recognized",
                "index": disease_idx
            }
        
        disease_key = self.idx_to_disease_name[disease_idx]
        disease_label = self.disease_labels.get(disease_key, {})
        
        return {
            "index": disease_idx,
            "key": disease_key,
            "name": disease_label.get("name", disease_key),
            "description": disease_label.get("description", "")
        }

    def get_severity_info(self, severity_idx: int) -> dict:
        """
        Get info for severity prediction.
        
        Args:
            severity_idx: Severity index (0=normal, 1=abnormal)
        
        Returns:
            Dict with name, description, level
        """
        severity_label = SEVERITY_LABELS.get(severity_idx, {})
        
        return {
            "index": severity_idx,
            "name": severity_label.get("name", "Unknown"),
            "description": severity_label.get("description", ""),
            "level": severity_label.get("severity_level", -1)
        }
    
    def get_stage_name(self, disease_key: str, stage_idx: Optional[int]) -> Optional[str]:
        """
        Get stage name for disease.
        
        Args:
            disease_key: Disease key (e.g., "Breast_cancer")
            stage_idx: Stage index
        
        Returns:
            Stage name or None if not applicable
        """
        if stage_idx is None:
            return None
        
        if disease_key not in self.stage_labels:
            return f"Stage {stage_idx}"
        
        stage_names = self.stage_labels[disease_key].get("names", {})
        return stage_names.get(stage_idx, f"Stage {stage_idx}")
    
    def get_stage_info(self, disease_key: str, stage_idx: Optional[int]) -> Optional[dict]:
       
        if stage_idx is None:
            return None

        # Stage head is global; decode by global index first.
        global_stage = GLOBAL_STAGE_LABELS.get(stage_idx)
        if global_stage is not None:
            disease_stage_info = self.stage_labels.get(global_stage["disease_key"], {})
            return {
                "index": stage_idx,
                "name": global_stage["name"],
                "description": disease_stage_info.get("description", "")
            }
        
        # Fallback for unknown stage indices
        if disease_key not in self.stage_labels:
            return {
                "index": stage_idx,
                "name": f"Stage {stage_idx}",
                "description": ""
            }
        
        stage_info = self.stage_labels[disease_key]
        stage_names = stage_info.get("names", {})
        stage_name = stage_names.get(stage_idx, f"Stage {stage_idx}")
        
        return {
            "index": stage_idx,
            "name": stage_name,
            "description": stage_info.get("description", "")
        }
    
    def map_prediction(self, prediction: dict) -> dict:
        
        disease_idx = prediction.get('disease_idx')
        disease_key_from_model = prediction.get('disease_name')
        severity_idx = prediction.get('severity_idx')
        stage_idx = prediction.get('stage_idx')

        # Single source of truth: disease_idx.
        disease_info = self.get_disease_info(disease_idx)
        disease_key = disease_info.get("key")

        # Strict safety validation: if upstream provides disease_name, it must match disease_idx mapping.
        if disease_key_from_model is not None and disease_key_from_model != disease_key:
            logger.error(
                "Disease mapping mismatch detected: disease_idx=%s maps to '%s' but model provided '%s'",
                disease_idx,
                disease_key,
                disease_key_from_model,
            )
            raise ValueError(
                f"Disease mapping mismatch: idx={disease_idx}, idx_key='{disease_key}', model_key='{disease_key_from_model}'"
            )

        severity_info = self.get_severity_info(severity_idx)
        stage_info = self.get_stage_info(disease_key, stage_idx) if severity_idx == 1 else None
        
        # Debug logging
        logger.info(f"\n{'='*70}")
        logger.info(f"🗂️  LABEL MAPPING RESULT:")
        logger.info(f"   Disease: idx={disease_idx} → {disease_info.get('name')}")
        logger.info(f"   Severity: idx={severity_idx} → {severity_info.get('name')}")
        if stage_idx is not None:
            if stage_info:
                logger.info(f"   Stage: idx={stage_idx} → {stage_info.get('name')}")
            else:
                logger.info(f"   Stage: idx={stage_idx} → NULL (not applicable for this disease)")
        logger.info(f"{'='*70}\n")
        
        # Build result dict
        result = {
            # Disease information
            "disease": {
                "index": disease_info.get("index"),
                "key": disease_info.get("key"),
                "name": disease_info.get("name"),
                "description": disease_info.get("description"),
                "confidence": prediction.get('disease_confidence', 0.0)
            },
            # Severity information
            "severity": {
                "index": severity_idx,
                "name": severity_info.get("name"),
                "description": severity_info.get("description"),
                "level": severity_info.get("level"),
                "confidence": prediction.get('severity_confidence', 0.0)
            },
            # Stage information (if abnormal and applicable)
            "stage": stage_info,
            # Overall diagnosis
            "diagnosis": f"{disease_info.get('name')} - {severity_info.get('name')}"
        }
        
        # Add stage to diagnosis if applicable
        if stage_info:
            result["diagnosis"] = f"{disease_info.get('name')} - {stage_info.get('name')}"
        
        # Include attention weights if available
        if 'attention_weights' in prediction:
            result["attention_weights"] = prediction['attention_weights']
        
        return result


def get_label_mapper(disease_names: list = None) -> LabelMapper:
    return LabelMapper(disease_names=disease_names or TRAINING_DISEASE_NAMES)


def update_label_mapper(disease_names: list):
    logger.info(f"Creating fresh label mapper with disease names: {disease_names}")
    return LabelMapper(disease_names=disease_names or TRAINING_DISEASE_NAMES)