import logging
from fastapi import APIRouter, HTTPException
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.core.enhanced_ai_service import get_ai_response
import os
from pathlib import Path

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.get("/chunk/{chunk_id}")
def get_chunk_text(chunk_id: str):
    """
    Search content_store for a file containing the chunk_id in its filename and return its text content.
    """
    content_store_dir = Path("./content_store")
    for root, dirs, files in os.walk(content_store_dir):
        for file in files:
            if chunk_id in file:
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return {"chunk_id": chunk_id, "text": f.read()}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error reading chunk file: {str(e)}")
    raise HTTPException(status_code=404, detail=f"Chunk with id {chunk_id} not found.")
# backend/api/v1/endpoints/chat.py
import logging
from fastapi import APIRouter
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.core.enhanced_ai_service import get_ai_response

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    """
    Handles a chat request by forwarding it to the AI service and
    returning the simplified response.
    """
    logger.info(f"Received chat request: {request.dict()}")
    
    ai_response_dict = get_ai_response(
        user_message=request.message,
        history=request.history,
        jurisdiction=request.jurisdiction
    )
    
    logger.info(f"Raw response from AI service: {ai_response_dict}")
    
    # The dictionary from the AI service now directly maps to the simplified ChatResponse
    response = ChatResponse(**ai_response_dict)
    
    logger.info(f"Sending final response to frontend: {response.dict()}")
    
    return response
