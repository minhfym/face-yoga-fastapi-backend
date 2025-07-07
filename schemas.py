from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime

# User schemas
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str

class UserLogin(BaseModel):
    username: str  # FastAPI OAuth2 expects 'username' field
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Client schemas
class ClientCreate(BaseModel):
    name: str
    email: Optional[str] = None
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
    analysis_type: str
    landmarks_detected: int
    results: Optional[Dict[str, Any]]
    created_at: datetime
    client_id: Optional[int]
    
    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
