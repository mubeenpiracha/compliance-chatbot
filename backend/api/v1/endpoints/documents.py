# backend/api/v1/endpoints/documents.py
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.schemas.regulatory_document import RegulatoryDocument, RegulatoryDocumentCreate
from backend.crud.crud_regulatory_document import create_document as create_doc_crud, get_documents as get_docs_crud
from backend.db.session import get_db

router = APIRouter()

@router.post("/", response_model=RegulatoryDocument)
def create_document(document: RegulatoryDocumentCreate, db: Session = Depends(get_db)):
    return create_doc_crud(db=db, document=document)

@router.get("/", response_model=List[RegulatoryDocument])
def read_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    documents = get_docs_crud(db, skip=skip, limit=limit)
    return documents
