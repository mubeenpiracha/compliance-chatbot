# backend/crud/crud_regulatory_document.py
from sqlalchemy.orm import Session
from backend.models.regulatory_document import RegulatoryDocument
from backend.schemas.regulatory_document import RegulatoryDocumentCreate

def get_document(db: Session, doc_id: int):
    """
    Fetches a single document by its ID.
    """
    return db.query(RegulatoryDocument).filter(RegulatoryDocument.doc_id == doc_id).first()

def get_documents(db: Session, skip: int = 0, limit: int = 100):
    """
    Fetches a list of documents with pagination.
    """
    return db.query(RegulatoryDocument).offset(skip).limit(limit).all()

def create_document(db: Session, document: RegulatoryDocumentCreate):
    """
    Creates a new document in the database.
    """
    # Create a new SQLAlchemy model instance from the Pydantic schema data
    db_document = RegulatoryDocument(
        title=document.title,
        jurisdiction=document.jurisdiction,
        source_url=document.source_url
    )
    # Add the new instance to the session
    db.add(db_document)
    # Commit the session to save the record to the database
    db.commit()
    # Refresh the instance to get the new data from the DB, like the generated doc_id
    db.refresh(db_document)
    return db_document
