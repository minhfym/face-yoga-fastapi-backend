from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List, Dict, Any

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    name: str
    role: str = "coach"

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    certified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Authentication schemas
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Client schemas
class ClientBase(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None

class ClientCreate(ClientBase):
    pass

class ClientResponse(ClientBase):
    id: int
    coach_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Analysis schemas
class AnalysisData(BaseModel):
    landmarks_count: int
    facial_regions: Dict[str, Any]
    wrinkle_analysis: Dict[str, Any]
    symmetry_analysis: Dict[str, Any]
    contour_analysis: Dict[str, Any]
    improvements: List[str]
    recommendations: List[str]

class AnalysisResponse(BaseModel):
    id: int
    client_id: int
    coach_id: int
    before_image_url: str
    after_image_url: str
    analysis_data: AnalysisData
    wrinkle_score: float
    symmetry_score: float
    contour_score: float
    overall_score: float
    notes: Optional[str]
    processing_time: float
    landmarks_detected: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class AnalysisRequest(BaseModel):
    client_id: int
    notes: Optional[str] = ""
