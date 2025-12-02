
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class StrategyInfo(BaseModel):
    """Information about a history strategy."""
    id: str
    name: str
    description: str
    settings: dict = {}


class StrategiesResponse(BaseModel):
    """Available history management strategies."""
    current: str
    strategies: list[StrategyInfo]


class ChangeStrategyRequest(BaseModel):
    """Request to change history strategy."""
    strategy: str


class ChangeStrategyResponse(BaseModel):
    """Response after changing strategy."""
    success: bool
    current_strategy: str
    message: str


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


STRATEGY_INFO = {
    "sliding_window": StrategyInfo(
        id="sliding_window",
        name="Sliding Window",
        description="Keeps the last N messages. Simple and fast, best for most use cases.",
        settings={"max_messages": 6}
    ),
    "token_budget": StrategyInfo(
        id="token_budget",
        name="Token Budget",
        description="Limits history by token count. Respects model context limits precisely.",
        settings={"max_tokens": 4096}
    ),
    "summarization": StrategyInfo(
        id="summarization",
        name="Summarization",
        description="Summarizes older messages, keeps recent ones. Best for long conversations.",
        settings={"summarize_after": 8, "summary_max_tokens": 500}
    ),
}
