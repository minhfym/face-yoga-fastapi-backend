from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import uvicorn
import os
from dotenv import load_dotenv

from database import get_db, engine
from models import Base
from auth import verify_token
from ml_analyzer import FaceYogaAnalyzer
from schemas import UserResponse, AnalysisResponse, LoginRequest, RegisterRequest

# Load environment variables
load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Face Yoga Analysis API",
    description="Advanced facial analysis using MediaPipe and OpenCV",
    version="1.0.0"
)

# CORS middleware - Updated for Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://localhost:3000", 
        "https://your-vercel-app.vercel.app",  # Add your Vercel URL here
        "*"  # For development only - remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize ML analyzer
analyzer = FaceYogaAnalyzer()

@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    try:
        await analyzer.initialize()
        print("🚀 Face Yoga Analysis API started successfully!")
        print("📊 MediaPipe and OpenCV models loaded")
    except Exception as e:
        print(f"❌ Error initializing ML models: {e}")

@app.get("/")
async def root():
    return {
        "message": "Face Yoga Analysis API",
        "version": "1.0.0",
        "status": "running",
        "ml_models": "MediaPipe + OpenCV",
        "platform": "Railway"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ml_analyzer": analyzer.is_initialized(),
        "database": "connected",
        "platform": "Railway"
    }

# Import route handlers
from routes import auth, analysis, clients

# Register routes
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(clients.router, prefix="/api/clients", tags=["Clients"])

# Railway deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
