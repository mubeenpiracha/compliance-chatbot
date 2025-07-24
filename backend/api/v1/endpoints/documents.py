# backend/api/v1/endpoints/documents.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import schemas, crud
from backend.db.session import get_db

# Create a new router instance
router = APIRouter()

@router.post("/", response_model=schemas.RegulatoryDocument)
def create_document(
    document: schemas.RegulatoryDocumentCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new regulatory document.
    """
    return crud.crud_regulatory_document.create_document(db=db, document=document)

@router.get("/", response_model=List[schemas.RegulatoryDocument])
def read_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Retrieve all regulatory documents.
    """
    documents = crud.crud_regulatory_document.get_documents(db, skip=skip, limit=limit)
    return documents
