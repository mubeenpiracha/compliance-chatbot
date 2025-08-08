# backend/schemas/chat.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str
    jurisdiction: str  # 'DIFC', 'ADGM', or 'Both'
    history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    sender: str
    text: str
    sources: Optional[List[Dict[str, Any]]] = []