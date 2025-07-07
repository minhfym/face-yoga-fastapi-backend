from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from database import get_db, engine
    from models import Base, User, Client, Analysis
    from schemas import UserCreate, UserLogin, ClientCreate, AnalysisResponse
    from auth import create_access_token, verify_token, get_password_hash, verify_password
    from ml_analyzer import get_analyzer
    logger.info("✅ All modules imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import modules: {e}")
    raise

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create database tables: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Face Yoga Analysis API",
    description="Advanced ML-powered facial analysis using MediaPipe and OpenCV",
    version="1.0.0"
)

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins + ["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize ML analyzer
try:
    ml_analyzer = get_analyzer()
    ML_ANALYZER_STATUS = ml_analyzer.is_initialized()
    logger.info(f"✅ ML Analyzer status: {ML_ANALYZER_STATUS}")
except Exception as e:
    logger.error(f"❌ Failed to initialize ML analyzer: {e}")
    ML_ANALYZER_STATUS = False

@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    try:
        if not ML_ANALYZER_STATUS:
            await ml_analyzer.initialize()
        logger.info("🚀 Face Yoga Analysis API started successfully!")
        logger.info("📊 MediaPipe and OpenCV models loaded")
    except Exception as e:
        logger.error(f"❌ Error initializing ML models: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Face Yoga Analysis API - Google Cloud Run",
        "platform": "Google Cloud Run",
        "ml_analyzer": ML_ANALYZER_STATUS,
        "version": "1.0.0",
        "status": "running",
        "features": ["MediaPipe 468 landmarks", "OpenCV analysis", "Real ML processing"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        db = next(get_db())
        db.execute("SELECT 1")
        db.close()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "database": db_status,
        "ml_analyzer": ML_ANALYZER_STATUS,
        "platform": "Google Cloud Run",
        "mediapipe": "enabled",
        "opencv": "enabled"
    }

# Authentication endpoints
@app.post("/api/auth/register")
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            email=user_data.email,
            name=user_data.name,
            hashed_password=hashed_password
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create access token
        access_token = create_access_token(data={"sub": new_user.email})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": new_user.id,
                "email": new_user.email,
                "name": new_user.name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login")
async def login(form_data: UserLogin, db: Session = Depends(get_db)):
    """Login user."""
    try:
        # Find user by email
        user = db.query(User).filter(User.email == form_data.username).first()
        
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create access token
        access_token = create_access_token(data={"sub": user.email})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

# Protected route helper
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Get current authenticated user."""
    try:
        payload = verify_token(credentials.credentials)
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# Client management endpoints
@app.post("/api/clients")
async def create_client(
    client_data: ClientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new client."""
    try:
        new_client = Client(
            name=client_data.name,
            email=client_data.email,
            phone=client_data.phone,
            notes=client_data.notes,
            user_id=current_user.id
        )
        
        db.add(new_client)
        db.commit()
        db.refresh(new_client)
        
        return {
            "id": new_client.id,
            "name": new_client.name,
            "email": new_client.email,
            "phone": new_client.phone,
            "notes": new_client.notes,
            "created_at": new_client.created_at
        }
        
    except Exception as e:
        logger.error(f"Client creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create client")

@app.get("/api/clients")
async def get_clients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all clients for the current user."""
    try:
        clients = db.query(Client).filter(Client.user_id == current_user.id).all()
        
        return [
            {
                "id": client.id,
                "name": client.name,
                "email": client.email,
                "phone": client.phone,
                "notes": client.notes,
                "created_at": client.created_at
            }
            for client in clients
        ]
        
    except Exception as e:
        logger.error(f"Failed to fetch clients: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch clients")

# Analysis endpoints
@app.post("/api/analysis/upload-and-analyze")
async def upload_and_analyze(
    before_image: UploadFile = File(...),
    after_image: Optional[UploadFile] = File(None),
    client_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload images and perform ML analysis."""
    try:
        if not ML_ANALYZER_STATUS:
            raise HTTPException(status_code=503, detail="ML analyzer not available")
        
        # Validate file types
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        
        if before_image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid before image format")
        
        if after_image and after_image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid after image format")
        
        # Read image data
        before_data = await before_image.read()
        
        if after_image:
            # Compare before and after images
            after_data = await after_image.read()
            analysis_result = await ml_analyzer.analyze_images(before_data, after_data)
        else:
            # Analyze single image
            analysis_result = await ml_analyzer.analyze_single_image(before_data)
        
        if not analysis_result.get("success", True):
            raise HTTPException(
                status_code=400, 
                detail=f"Analysis failed: {analysis_result.get('error', 'Unknown error')}"
            )
        
        # Save analysis to database
        new_analysis = Analysis(
            user_id=current_user.id,
            client_id=client_id,
            analysis_type="comparison" if after_image else "single",
            results=analysis_result,
            landmarks_detected=analysis_result.get("landmarks_count", 0)
        )
        
        db.add(new_analysis)
        db.commit()
        db.refresh(new_analysis)
        
        return {
            "success": True,
            "analysis_id": new_analysis.id,
            "analysis_type": new_analysis.analysis_type,
            "landmarks_detected": analysis_result.get("landmarks_count", 0),
            "results": analysis_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analyses")
async def get_analyses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all analyses for the current user."""
    try:
        analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).all()
        
        return [
            {
                "id": analysis.id,
                "client_id": analysis.client_id,
                "analysis_type": analysis.analysis_type,
                "landmarks_detected": analysis.landmarks_detected,
                "created_at": analysis.created_at,
                "results": analysis.results
            }
            for analysis in analyses
        ]
        
    except Exception as e:
        logger.error(f"Failed to fetch analyses: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analyses")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
