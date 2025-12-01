
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Citation(BaseModel):
    """Citation information from retrieved documents."""
    number: int
    title: str
    display: str
    source_type: str
    content_preview: str
    url: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[int] = None


class MessageResponse(BaseModel):
    """Individual message in history."""
    role: str
    content: str
    timestamp: Optional[str] = None
    citations: Optional[list[Citation]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    citations: list[Citation] = []


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]


class StatusResponse(BaseModel):
    status: str
    model: str
    llm_url: str
    vector_store: str
    collection: str


class ConfigResponse(BaseModel):
    ollama_model: str
    ollama_base_url: str
    chroma_collection: str
    embedding_model: str
    retrieval_k: int
    history_strategy: str
    history_max_messages: int


class SessionInfo(BaseModel):
    """Information about a single chat session."""
    session_id: str
    message_count: int
    created_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    preview: Optional[str] = Field(
        None, description="Preview of the last message")


class SessionListResponse(BaseModel):
    """Response containing all active sessions."""
    total_sessions: int
    sessions: list[SessionInfo]
