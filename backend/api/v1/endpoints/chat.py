# backend/api/v1/endpoints/chat.py
from fastapi import APIRouter
from backend.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    """
    This endpoint receives a user's message and returns a dummy AI response.
    In a future phase, this is where the RAG pipeline will be called.
    """
    # For now, we just echo the message back.
    # This proves the end-to-end connection is working.
    ai_response_text = f"You said: '{request.message}'. I am a simple echo bot for now."

    # Return the response in the format defined by our ChatResponse schema
    return ChatResponse(sender="ai", text=ai_response_text)
