"""
Database connection and session management.
Handles SQLAlchemy engine and session creation.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create database engine
# Echo=True shows SQL queries in console (useful for debugging)
engine = create_engine(
    settings.database_url,
    echo=True if settings.environment == "development" else False,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,  # Number of connections to keep
    max_overflow=10  # Maximum overflow connections
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for all database models
Base = declarative_base()


def get_db():
    """
    Dependency that provides database session to route handlers.
    Automatically closes session after request completes.
    
    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db here
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()