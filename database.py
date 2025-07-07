from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging

logger = logging.getLogger(__name__)

# Get database URL from environment
NEON_DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Fallback for local development
    DATABASE_URL = "postgresql://user:password@localhost/faceYoga"
    logger.warning("Using fallback DATABASE_URL for local development")

# Create engine
try:
    engine = create_engine(DATABASE_URL)
    logger.info("✅ Database engine created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create database engine: {e}")
    raise

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
