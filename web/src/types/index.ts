export interface Message {
    role: 'user' | 'assistant';
    content: string;
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
}

export interface HistoryResponse {
    session_id: string;
    messages: Message[];
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
}