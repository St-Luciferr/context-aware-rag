import type {
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    SessionListResponse,
    StatusResponse,
    ConfigResponse,
    StrategiesResponse,
    ChangeStrategyRequest,
    ChangeStrategyResponse,
    TopicsResponse,
    AddTopicRequest,
    AddTopicResponse,
    RemoveTopicResponse,
    IngestResponse,
} from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiError extends Error {
    constructor(public status: number, message: string) {
        super(message);
        this.name = 'ApiError';
    }
}

async function fetchApi<T>(
    endpoint: string,
    options?: RequestInit
): Promise<T> {
    const url = `${API_BASE}${endpoint}`;

    try {
        const res = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options?.headers,
            },
        });

        if (!res.ok) {
            const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
            throw new ApiError(res.status, error.detail || `Request failed: ${res.status}`);
        }

        return res.json();
    } catch (err) {
        if (err instanceof ApiError) {
            throw err;
        }
        // Network error or other fetch failure
        throw new ApiError(0, 'Failed to connect to server. Please check if the API is running.');
    }
}

export const api = {
    // Status
    getStatus: () => fetchApi<StatusResponse>('/api/v1/status'),

    getConfig: () => fetchApi<ConfigResponse>('/api/v1/config'),

    // Sessions
    getSessions: () => fetchApi<SessionListResponse>('/api/v1/sessions'),

    createSession: () =>
        fetchApi<{ session_id: string }>('/api/v1/session/new', { method: 'POST' }),

    deleteSession: (sessionId: string) =>
        fetchApi<{ message: string }>(`/api/v1/session/${sessionId}`, {
            method: 'DELETE',
        }),

    getHistory: (sessionId: string, limit: number = 20, before?: string) => {
        const params = new URLSearchParams({ limit: String(limit) });
        if (before) params.append('before', before);
        return fetchApi<HistoryResponse>(`/api/v1/history/${sessionId}?${params}`);
    },

    // Chat
    sendMessage: (data: ChatRequest) =>
        fetchApi<ChatResponse>('/api/v1/chat', {
            method: 'POST',
            body: JSON.stringify(data),
        }),

    // History Strategies
    getStrategies: () => fetchApi<StrategiesResponse>('/api/v1/strategies'),

    changeStrategy: (data: ChangeStrategyRequest) =>
        fetchApi<ChangeStrategyResponse>('/api/v1/strategies', {
            method: 'POST',
            body: JSON.stringify(data),
        }),

    // Topics Management
    getTopics: () => fetchApi<TopicsResponse>('/api/v1/topics'),

    addTopic: (data: AddTopicRequest) =>
        fetchApi<AddTopicResponse>('/api/v1/topics', {
            method: 'POST',
            body: JSON.stringify(data),
        }),

    removeTopic: (topic: string) =>
        fetchApi<RemoveTopicResponse>(`/api/v1/topics/${encodeURIComponent(topic)}`, {
            method: 'DELETE',
        }),

    resetTopics: () =>
        fetchApi<AddTopicResponse>('/api/v1/topics/reset', {
            method: 'POST',
        }),

    // Incremental ingestion - only ingest pending topics
    ingestPendingTopics: () =>
        fetchApi<IngestResponse>('/api/v1/topics/ingest', {
            method: 'POST',
        }),

    // Full re-ingestion - delete and re-ingest all topics
    fullReingest: () =>
        fetchApi<IngestResponse>('/api/v1/topics/ingest/full', {
            method: 'POST',
        }),
};

export { ApiError };