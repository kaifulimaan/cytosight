"""
Initialize database tables.
Run this once to create all tables in the database.
"""

from app.database.database import engine, Base
from app.database.models import User, Diagnosis, Segmentation


def init_db():
    """Create all tables in the database."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")


if __name__ == "__main__":
    init_db()