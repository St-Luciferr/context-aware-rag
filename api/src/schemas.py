
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
    has_more: bool = False
    total_count: int = 0
    oldest_timestamp: Optional[str] = None


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


# ==================== Topics Schemas ====================

class TopicInfo(BaseModel):
    """Information about a single topic."""
    name: str
    is_default: bool
    is_ingested: bool


class TopicsResponse(BaseModel):
    """Response containing current Wikipedia topics with status."""
    topics: list[TopicInfo]
    total: int
    ingested_count: int
    pending_count: int


class AddTopicRequest(BaseModel):
    """Request to add a new Wikipedia topic."""
    topic: str


class AddTopicResponse(BaseModel):
    """Response after adding a topic."""
    success: bool
    message: str


class RemoveTopicResponse(BaseModel):
    """Response after removing a topic."""
    success: bool
    message: str


class IngestResponse(BaseModel):
    """Response from ingestion operation."""
    status: str
    message: str
    document_count: Optional[int] = None
    chunks_added: Optional[int] = None
    topics_added: Optional[list[str]] = None


# ==================== Evaluation Schemas ====================

class DatasetGenerateRequest(BaseModel):
    """Request to generate an evaluation dataset."""
    name: str = Field(..., description="Name for the dataset")
    questions_per_topic: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of questions to generate per topic"
    )
    question_types: list[str] = Field(
        default=["factual", "comparative", "explanatory"],
        description="Types of questions to generate"
    )


class DatasetInfo(BaseModel):
    """Information about an evaluation dataset."""
    name: str
    created_at: Optional[str] = None
    question_count: int
    topics: list[str]


class DatasetListResponse(BaseModel):
    """Response containing available evaluation datasets."""
    datasets: list[DatasetInfo]
    total: int


class DatasetGenerateResponse(BaseModel):
    """Response after generating a dataset."""
    success: bool
    name: str
    question_count: int
    topics_covered: list[str]
    message: str


class EvalRunRequest(BaseModel):
    """Request to run an evaluation."""
    dataset_name: str = Field(..., description="Name of the dataset to evaluate")
    experiment_name: Optional[str] = Field(
        None,
        description="Optional name for this evaluation experiment"
    )


class MetricSummary(BaseModel):
    """Summary statistics for a metric."""
    mean: float
    std: float
    min: float
    max: float


class EvalRunResponse(BaseModel):
    """Response after running an evaluation."""
    success: bool
    run_id: str
    dataset_name: str
    total_questions: int
    successful_questions: int
    failed_questions: int
    total_time_seconds: float
    message: str


class EvalResultInfo(BaseModel):
    """Summary info about an evaluation result."""
    run_id: str
    dataset_name: Optional[str] = None
    timestamp: Optional[str] = None
    total_questions: int
    successful_questions: int
    model: Optional[str] = None


class EvalResultsListResponse(BaseModel):
    """Response containing available evaluation results."""
    results: list[EvalResultInfo]
    total: int


class EvalSummaryResponse(BaseModel):
    """Detailed summary of an evaluation run."""
    run_id: str
    dataset: str
    model: Optional[str] = None
    timestamp: Optional[str] = None
    questions: dict
    timing: dict
    retrieval: Optional[dict] = None
    generation: Optional[dict] = None


class EvalReportResponse(BaseModel):
    """Response containing path to generated report."""
    success: bool
    run_id: str
    report_path: str
    message: str


class EvalCompareRequest(BaseModel):
    """Request to compare multiple evaluation runs."""
    run_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of run IDs to compare"
    )
