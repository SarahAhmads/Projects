from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    sender: str
    text: str

class ChatQuery(BaseModel):
    user_id: str
    message: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    status: str = "success"

class SettingsModel(BaseModel):
    systemPrompt: str
    temperature: float
    model: str