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
