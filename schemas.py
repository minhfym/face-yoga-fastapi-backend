from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime

# User schemas
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str

class UserLogin(BaseModel):
    username: EmailStr  # FastAPI OAuth2 expects 'username' field
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Client schemas
class ClientCreate(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    notes: Optional[str] = None

class ClientResponse(BaseModel):
    id: int
    name: str
    email: Optional[str]
    phone: Optional[str]
    notes: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

# Analysis schemas
class AnalysisResponse(BaseModel):
    id: int
    user_id: int
    client_id: Optional[int]
    analysis_type: str
    results: Dict[str, Any]
    landmarks_detected: int
    created_at: datetime
    
    class Config:
        from_attributes = True
