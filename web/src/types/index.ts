// Types for the RAG Chatbot API

export interface Citation {
    number: number;
    title: string;
    display: string;
    source_type: string;
    content_preview: string;
    url?: string;
    page?: number;
    chunk_id?: number;
}

export interface Message {
    id?: number;
    role: 'user' | 'assistant';
    content: string;
    timestamp?: string;
    citations?: Citation[];
}

export interface Session {
    session_id: string;
    message_count: number;
    created_at: string | null;
    last_message_at: string | null;
    preview: string | null;
}

export interface SessionListResponse {
    total_sessions: number;
    sessions: Session[];
}

export interface ChatRequest {
    message: string;
    session_id?: string;
}

export interface ChatResponse {
    response: string;
    session_id: string;
    citations: Citation[];
}

export interface HistoryResponse {
    session_id: string;
    messages: Message[];
    has_more: boolean;
    total_count: number;
    oldest_timestamp: string | null;
}

export interface StatusResponse {
    status: string;
    model: string;
    vector_store: string;
    collection: string;
}

export interface ConfigResponse {
    ollama_model: string;
    ollama_base_url: string;
    chroma_collection: string;
    embedding_model: string;
    retrieval_k: number;
    history_strategy?: string;
    history_max_messages?: number;
}

// History Strategy Types
export interface StrategyInfo {
    id: string;
    name: string;
    description: string;
    settings: Record<string, number>;
}

export interface StrategiesResponse {
    current: string;
    strategies: StrategyInfo[];
}

export interface ChangeStrategyRequest {
    strategy: string;
}

export interface ChangeStrategyResponse {
    success: boolean;
    current_strategy: string;
    message: string;
}

// Topics Types
export interface TopicInfo {
    name: string;
    is_default: boolean;
    is_ingested: boolean;
}

export interface TopicsResponse {
    topics: TopicInfo[];
    total: number;
    ingested_count: number;
    pending_count: number;
}

export interface AddTopicRequest {
    topic: string;
}

export interface AddTopicResponse {
    success: boolean;
    message: string;
}

export interface RemoveTopicResponse {
    success: boolean;
    message: string;
}

export interface IngestResponse {
    status: string;
    message: string;
    document_count?: number;
    chunks_added?: number;
    topics_added?: string[];
}