import type {
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    SessionListResponse,
    StatusResponse,
    ConfigResponse,
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
    const res = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
        },
    });

    if (!res.ok) {
        const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new ApiError(res.status, error.detail || 'Request failed');
    }

    return res.json();
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

    getHistory: (sessionId: string) =>
        fetchApi<HistoryResponse>(`/api/v1/history/${sessionId}`),

    // Chat
    sendMessage: (data: ChatRequest) =>
        fetchApi<ChatResponse>('/api/v1/chat', {
            method: 'POST',
            body: JSON.stringify(data),
        }),
};

export { ApiError };