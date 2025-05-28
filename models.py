from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="coach")  # coach, admin
    certified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    clients = relationship("Client", back_populates="coach")
    analyses = relationship("Analysis", back_populates="coach")

class Client(Base):
    __tablename__ = "clients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    phone = Column(String, nullable=True)
    coach_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    coach = relationship("User", back_populates="clients")
    analyses = relationship("Analysis", back_populates="client")

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    coach_id = Column(Integer, ForeignKey("users.id"))
    before_image_url = Column(String, nullable=False)
    after_image_url = Column(String, nullable=False)
    
    # Analysis results (JSON)
    analysis_data = Column(JSON, nullable=False)
    
    # Scores
    wrinkle_score = Column(Float)
    symmetry_score = Column(Float)
    contour_score = Column(Float)
    overall_score = Column(Float)
    
    # Additional data
    notes = Column(Text)
    processing_time = Column(Float)
    landmarks_detected = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    client = relationship("Client", back_populates="analyses")
    coach = relationship("User", back_populates="analyses")
