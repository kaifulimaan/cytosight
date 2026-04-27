"""
Database initialization script.

Run this once to create all necessary tables in Supabase.

Usage:
    python init_db.py
"""

import logging
import sys
from app.database.database import engine, Base
from app.database import models  # Import all models to register them

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db():
    """
    Create all database tables defined in models.py
    
    This is idempotent - safe to run multiple times.
    """
    try:
        logger.info("\n" + "="*80)
        logger.info("🗄️  Initializing Database Tables...")
        logger.info("="*80)
        
        # Create all tables based on Base metadata
        logger.info("Creating tables if they don't exist...")
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Database initialized successfully!")
        logger.info("\nTables created:")
        logger.info("  - users")
        logger.info("  - diagnoses")
        logger.info("  - segmentations")
        
        logger.info("\n" + "="*80)
        logger.info("Next steps:")
        logger.info("1. Start backend: uvicorn app.main:app --reload --port 8000")
        logger.info("2. Visit Swagger: http://localhost:8000/docs")
        logger.info("3. Login to Loveable: https://cytosight.lovable.app/login")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Database initialization failed: {e}", exc_info=True)
        logger.error("\nTroubleshooting:")
        logger.error("1. Check DATABASE_URL in .env")
        logger.error("2. Verify Supabase credentials are correct")
        logger.error("3. Ensure PostgreSQL is accessible")
        logger.error("4. Check internet connection")
        return False


if __name__ == "__main__":
    success = init_db()
    sys.exit(0 if success else 1)
