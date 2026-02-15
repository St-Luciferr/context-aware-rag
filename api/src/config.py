"""
Centralized Configuration using Pydantic Settings
Loads environment variables from .env file with validation
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class HistorySettings(BaseSettings):
    """Conversation history management configuration."""
    # Strategy: sliding_window, token_budget, summarization
    strategy: str = Field(
        default="sliding_window",
        alias="HISTORY_STRATEGY",
        description="History management strategy"
    )

    # Sliding window settings
    max_messages: int = Field(
        default=10,
        ge=2,
        le=100,
        alias="HISTORY_MAX_MESSAGES"
    )

    # Token budget settings
    max_tokens: int = Field(
        default=4000,
        ge=500,
        le=32000,
        alias="HISTORY_MAX_TOKENS"
    )

    # Summarization settings
    summarize_after: int = Field(
        default=8,
        ge=4,
        le=50,
        alias="HISTORY_SUMMARIZE_AFTER",
        description="Summarize messages older than this count"
    )
    summary_max_tokens: int = Field(
        default=500,
        ge=100,
        le=2000,
        alias="HISTORY_SUMMARY_MAX_TOKENS"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class OllamaSettings(BaseSettings):
    """Ollama LLM configuration."""
    model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    base_url: str = Field(default="http://localhost:11434",
                          alias="OLLAMA_BASE_URL")
    api_key: str = Field(default="", alias="OLLAMA_API_KEY")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, alias="OLLAMA_TEMPERATURE")
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class ChromaSettings(BaseSettings):
    """ChromaDB vector store configuration."""
    persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(
        default="wiki_docs", alias="CHROMA_COLLECTION_NAME")
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    model_name: str = Field(default="all-MiniLM-L6-v2",
                            alias="EMBEDDING_MODEL")
    device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    normalize: bool = True
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class RAGSettings(BaseSettings):
    """RAG pipeline configuration."""
    retrieval_k: int = Field(default=5, ge=1, le=20, alias="RAG_RETRIEVAL_K")
    distilled_retrieval_k: int = Field(
        default=3, ge=1, le=5, alias="DISTILLED_RETRIEVAL_K")
    # Semantic chunker settings
    breakpoint_threshold_type: str = Field(
        default="percentile",
        alias="RAG_BREAKPOINT_TYPE",
        description="Type: percentile, standard_deviation, interquartile, gradient"
    )
    breakpoint_threshold_amount: float = Field(
        default=95.0,
        alias="RAG_BREAKPOINT_AMOUNT",
        description="Threshold amount for semantic chunking"
    )


class APISettings(BaseSettings):
    """FastAPI server configuration."""
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, ge=1, le=65535, alias="API_PORT")
    title: str = "RAG Chatbot API"
    version: str = "1.0.0"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class LangSmithSettings(BaseSettings):
    """LangSmith tracing and observability configuration."""
    enabled: bool = Field(
        default=False,
        alias="LANGSMITH_TRACING",
        description="Enable LangSmith tracing (set to 'true' to enable)"
    )
    api_key: str = Field(
        default="",
        alias="LANGSMITH_API_KEY",
        description="LangSmith API key from smith.langchain.com"
    )
    project: str = Field(
        default="context-aware-rag",
        alias="LANGSMITH_PROJECT",
        description="LangSmith project name for grouping traces"
    )
    endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT",
        description="LangSmith API endpoint"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def is_configured(self) -> bool:
        """Check if LangSmith is properly configured and enabled."""
        return self.enabled and bool(self.api_key)


class EvaluationSettings(BaseSettings):
    """Evaluation pipeline configuration."""

    # Dataset generation
    questions_per_topic: int = Field(
        default=5,
        ge=1,
        le=50,
        alias="EVAL_QUESTIONS_PER_TOPIC",
        description="Number of questions to generate per topic"
    )
    default_question_types: list[str] = Field(
        default=["factual", "comparative", "explanatory"],
        alias="EVAL_QUESTION_TYPES",
        description="Types of questions to generate"
    )

    # Retrieval evaluation
    retrieval_k_values: list[int] = Field(
        default=[1, 3, 5, 10],
        alias="EVAL_RETRIEVAL_K_VALUES",
        description="K values for Precision@K and Recall@K"
    )

    # Generation evaluation thresholds
    faithfulness_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="EVAL_FAITHFULNESS_THRESHOLD"
    )
    relevance_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="EVAL_RELEVANCE_THRESHOLD"
    )

    # Persistence directories
    datasets_dir: str = Field(
        default="data/eval_datasets",
        alias="EVAL_DATASETS_DIR"
    )
    results_dir: str = Field(
        default="data/eval_results",
        alias="EVAL_RESULTS_DIR"
    )
    reports_dir: str = Field(
        default="data/eval_reports",
        alias="EVAL_REPORTS_DIR"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class Settings(BaseSettings):
    """Main settings class combining all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Nested settings
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    api: APISettings = Field(default_factory=APISettings)
    history: HistorySettings = Field(default_factory=HistorySettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    env: str = Field(default='dev', alias="ENV")
    workers: int = Field(default=1, alias="WORKERS")

    # Wikipedia topics for ingestion
    wiki_topics: list[str] = Field(
        default=[
            "Artificial intelligence",
            "Machine learning",
            "Natural language processing",
            "Deep learning",
            "Neural network"
        ],
        alias="WIKI_TOPICS"
    )

    @classmethod
    def parse_wiki_topics(cls, v):
        """Parse comma-separated topics string into list."""
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v


settings = Settings()
