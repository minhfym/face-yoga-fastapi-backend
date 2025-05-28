from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models import User, Client
from schemas import ClientCreate, ClientResponse
from auth import get_current_user

router = APIRouter()

@router.post("/", response_model=ClientResponse)
async def create_client(
    client_data: ClientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new client"""
    
    # Check if client email already exists
    existing_client = db.query(Client).filter(Client.email == client_data.email).first()
    if existing_client:
        raise HTTPException(status_code=400, detail="Client email already exists")
    
    db_client = Client(
        **client_data.dict(),
        coach_id=current_user.id
    )
    
    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    
    return ClientResponse.from_orm(db_client)

@router.get("/", response_model=List[ClientResponse])
async def get_clients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all clients for current coach"""
    
    clients = db.query(Client).filter(Client.coach_id == current_user.id).all()
    return [ClientResponse.from_orm(client) for client in clients]

@router.get("/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific client"""
    
    client = db.query(Client).filter(
        Client.id == client_id,
        Client.coach_id == current_user.id
    ).first()
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return ClientResponse.from_orm(client)

@router.put("/{client_id}", response_model=ClientResponse)
async def update_client(
    client_id: int,
    client_data: ClientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update client information"""
    
    client = db.query(Client).filter(
        Client.id == client_id,
        Client.coach_id == current_user.id
    ).first()
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    for field, value in client_data.dict().items():
        setattr(client, field, value)
    
    db.commit()
    db.refresh(client)
    
    return ClientResponse.from_orm(client)

@router.delete("/{client_id}")
async def delete_client(
    client_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete client"""
    
    client = db.query(Client).filter(
        Client.id == client_id,
        Client.coach_id == current_user.id
    ).first()
    
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    db.delete(client)
    db.commit()
    
    return {"message": "Client deleted successfully"}
