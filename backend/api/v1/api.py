# backend/api/v1/api.py
from fastapi import APIRouter
from backend.api.v1.endpoints import documents

# This is the main router for the v1 API
api_router = APIRouter()

# Include the documents router
# All routes from documents.py will now be prefixed with '/documents'
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
