# backend/api/v1/endpoints/chat.py
from fastapi import APIRouter
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.core.ai_service import get_ai_response # 1. Import our new AI service

router = APIRouter()

@router.post("/", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    """
    This endpoint receives a user's message, gets a response from the AI service,
    and returns it.
    """
    # 2. Call the AI service instead of echoing
    ai_response_text = get_ai_response(user_message=request.message)

    # 3. Return the AI's response
    return ChatResponse(sender="ai", text=ai_response_text)
