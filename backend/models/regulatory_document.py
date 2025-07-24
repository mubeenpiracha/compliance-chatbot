# backend/models/regulatory_document.py
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Boolean, ForeignKey, Date, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from backend.db.base import Base

class RegulatoryDocument(Base):
    __tablename__ = 'regulatory_documents'
    doc_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    jurisdiction = Column(String(10), nullable=False) # 'DIFC' or 'ADGM'
    source_url = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    versions = relationship("DocumentVersion", back_populates="document")


class DocumentVersion(Base):
    __tablename__ = 'document_versions'
    version_id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(Integer, ForeignKey('regulatory_documents.doc_id'), nullable=False)
    version_string = Column(String(50), nullable=False)
    effective_date = Column(Date, nullable=False)
    ingestion_date = Column(TIMESTAMP(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
    source_pdf_path = Column(Text, nullable=False)

    document = relationship("RegulatoryDocument", back_populates="versions")
    chunks = relationship("DocumentChunk", back_populates="version")


class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id = Column(Integer, ForeignKey('document_versions.version_id'), nullable=False)
    vector_id = Column(String(255), nullable=False, unique=True)
    paragraph_id = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    version = relationship("DocumentVersion", back_populates="chunks")


class IngestionLog(Base):
    __tablename__ = 'ingestion_log'
    log_id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey('document_versions.version_id'), nullable=True)
    status = Column(String(50), nullable=False) # 'SUCCESS', 'FAILURE', 'IN_PROGRESS'
    notes = Column(Text)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now())
