# backend/schemas/regulatory_document.py
from pydantic import BaseModel
from typing import Optional

class RegulatoryDocumentBase(BaseModel):
    title: str
    jurisdiction: str
    source_url: Optional[str] = None

class RegulatoryDocumentCreate(RegulatoryDocumentBase):
    pass

class RegulatoryDocument(RegulatoryDocumentBase):
    doc_id: int
    class Config:
        from_attributes = True # Corrected from orm_mode
