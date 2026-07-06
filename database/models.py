# database/models.py
# Purpose: Define database table structure
# SQLite stores every inspection permanently
# so factory managers can review history anytime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database file location
DATABASE_URL = "sqlite:///database/inspections.db"

# Base class for all table models
Base = declarative_base()

class Inspection(Base):
    """
    One row = one wafer inspection.
    Every time someone uploads an image and gets a prediction,
    we save a record here permanently.
    """
    __tablename__ = "inspections"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Image info
    filename = Column(String(255), nullable=False)
    
    # Prediction results
    defect_type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(String(50), nullable=False)
    pass_fail = Column(String(10), nullable=False)
    
    # Performance
    inference_time = Column(Float, nullable=False)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50), default="YOLOv8n-cls-v1")
    notes = Column(Text, nullable=True)


def get_engine():
    """Create database engine."""
    os.makedirs("database", exist_ok=True)
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    return engine


def create_tables():
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")
    return engine


def get_session():
    """Get a database session."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


# Initialize database when this file is run directly
if __name__ == "__main__":
    create_tables()
    print("✅ Database initialized successfully!")
    print(f"📁 Location: database/inspections.db")