# backend/api/v1/endpoints/chat.py
from fastapi import APIRouter
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.core.ai_service import get_ai_response

router = APIRouter()

@router.post("/", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    ai_response = get_ai_response(
        user_message=request.message,
        history=request.history,
        jurisdiction=request.jurisdiction
    )
    
    # Convert Document objects to dictionaries for Pydantic serialization
    sources = []
    if ai_response.get('sources'):
        for source in ai_response['sources']:
            if hasattr(source, 'page_content') and hasattr(source, 'metadata'):
                # This is a Document object - convert to dict
                sources.append({
                    'page_content': source.page_content,
                    'metadata': source.metadata
                })
            elif isinstance(source, dict):
                # Already a dictionary
                sources.append(source)
    
    return ChatResponse(
        sender="ai", 
        text=ai_response['answer'], 
        sources=sources
    )
