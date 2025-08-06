# backend/api/v1/endpoints/chat.py
from fastapi import APIRouter
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.core.ai_service import get_ai_response

router = APIRouter()

@router.post("/", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    ai_response_text = get_ai_response(
        user_message=request.message,
        history=request.history,
        jurisdiction=request.jurisdiction
    )
    return ChatResponse(sender="ai", text=ai_response_text)
