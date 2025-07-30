# backend/schemas/chat.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    """
    Schema for the request body of the /chat endpoint.
    """
    message: str
    # In the future, we can add conversation history here
    # history: list = []

class ChatResponse(BaseModel):
    """
    Schema for the response body of the /chat endpoint.
    """
    sender: str
    text: str