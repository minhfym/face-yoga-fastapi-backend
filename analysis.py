from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import asyncio

from database import get_db
from models import User, Client, Analysis
from schemas import AnalysisResponse, AnalysisRequest
from auth import get_current_user
from ml_analyzer import FaceYogaAnalyzer

router = APIRouter()

# Global analyzer instance
analyzer = FaceYogaAnalyzer()

@router.post("/upload-and-analyze", response_model=AnalysisResponse)
async def upload_and_analyze(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    client_id: int = Form(...),
    notes: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload images and perform ML analysis"""
    
    # Validate file types
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if before_image.content_type not in allowed_types or after_image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are allowed")
    
    # Verify client belongs to coach
    client = db.query(Client).filter(
        Client.id == client_id,
        Client.coach_id == current_user.id
    ).first()
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    try:
        # Read image bytes
        before_bytes = await before_image.read()
        after_bytes = await after_image.read()
        
        # Perform ML analysis
        if not analyzer.is_initialized():
            await analyzer.initialize()
        
        analysis_result = await analyzer.analyze_images(before_bytes, after_bytes)
        
        # Save analysis to database
        db_analysis = Analysis(
            client_id=client_id,
            coach_id=current_user.id,
            before_image_url=f"temp_before_{before_image.filename}",  # In production, save to cloud storage
            after_image_url=f"temp_after_{after_image.filename}",
            analysis_data=analysis_result,
            wrinkle_score=analysis_result['scores']['wrinkle_score'],
            symmetry_score=analysis_result['scores']['symmetry_score'],
            contour_score=analysis_result['scores']['contour_score'],
            overall_score=analysis_result['scores']['overall_score'],
            notes=notes,
            processing_time=analysis_result['analysis_metadata']['processing_time'],
            landmarks_detected=analysis_result['landmarks_count']
        )
        
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)
        
        return AnalysisResponse.from_orm(db_analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/history/{client_id}", response_model=List[AnalysisResponse])
async def get_analysis_history(
    client_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get analysis history for a client"""
    
    # Verify client belongs to coach
    client = db.query(Client).filter(
        Client.id == client_id,
        Client.coach_id == current_user.id
    ).first()
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    analyses = db.query(Analysis).filter(
        Analysis.client_id == client_id
    ).order_by(Analysis.created_at.desc()).all()
    
    return [AnalysisResponse.from_orm(analysis) for analysis in analyses]

@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific analysis"""
    
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.coach_id == current_user.id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return AnalysisResponse.from_orm(analysis)
