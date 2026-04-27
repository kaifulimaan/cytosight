"""
Database models (tables) for the application.
Uses SQLAlchemy ORM to define table structure.
"""

from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Float, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
from app.database.database import Base


class User(Base):
    """
    User table stores user authentication information.
    
    Columns:
        id: Unique user identifier (UUID)
        full_name: User's full name
        email: User's email (unique, used for login)
        password_hash: Hashed password (never store plain passwords!)
        created_at: When user account was created
        last_login: Last successful login timestamp
    """
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    diagnoses = relationship("Diagnosis", back_populates="user", cascade="all, delete-orphan")
    segmentations = relationship("Segmentation", back_populates="user", cascade="all, delete-orphan")


class Diagnosis(Base):
    """
    Diagnosis table stores disease diagnosis results.
    
    Columns:
        id: Unique diagnosis identifier
        user_id: Foreign key to users table
        disease_name: Predicted disease (e.g., "Breast Cancer")
        severity: Normal or Abnormal
        stage: Disease stage (e.g., "Ductal Carcinoma")
        confidence_*: Model confidence scores (0-1)
        *_image_url: S3 URLs for images (NOT actual image data)
        explainability_text: AI-generated explanation
        created_at: When diagnosis was performed
    """
    __tablename__ = "diagnoses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Diagnosis results
    disease_name = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    stage = Column(String(100), nullable=True)
    
    # Confidence scores
    confidence_disease = Column(Float, nullable=True)
    confidence_severity = Column(Float, nullable=True)
    confidence_stage = Column(Float, nullable=True)
    
    # Image URLs (S3 paths, not actual images!)
    original_image_url = Column(Text, nullable=False)
    heatmap_url = Column(Text, nullable=True)
    
    # Explainability
    explainability_text = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    user = relationship("User", back_populates="diagnoses")


class Segmentation(Base):
    """
    Segmentation table stores cell segmentation results.
    
    Columns:
        id: Unique segmentation identifier
        user_id: Foreign key to users table
        original_image_url: S3 URL for original image
        segmented_mask_url: S3 URL for segmentation mask
        cell_count: Number of cells detected
        created_at: When segmentation was performed
    """
    __tablename__ = "segmentations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Image URLs (S3 paths)
    original_image_url = Column(Text, nullable=False)
    segmented_mask_url = Column(Text, nullable=False)
    
    # Results
    cell_count = Column(Integer, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    user = relationship("User", back_populates="segmentations")