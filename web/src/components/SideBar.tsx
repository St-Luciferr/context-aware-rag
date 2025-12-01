'use client';

import { useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import {
    MessageSquare,
    Plus,
    Trash2,
    Menu,
    X,
    Settings,
    Loader2,
} from 'lucide-react';
import clsx from 'clsx';
import type { Session, StatusResponse } from '@/types';

interface SidebarProps {
    sessions: Session[];
    activeSessionId: string | null;
    status: StatusResponse | null;
    isLoading: boolean;
    onSelectSession: (sessionId: string) => void;
    onNewSession: () => void;
    onDeleteSession: (sessionId: string) => void;
}

export default function Sidebar({
    sessions,
    activeSessionId,
    status,
    isLoading,
    onSelectSession,
    onNewSession,
    onDeleteSession,
}: SidebarProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [deletingId, setDeletingId] = useState<string | null>(null);

    const handleDelete = async (e: React.MouseEvent, sessionId: string) => {
        e.stopPropagation();
        setDeletingId(sessionId);
        await onDeleteSession(sessionId);
        setDeletingId(null);
    };

    const formatTime = (dateStr: string | null) => {
        if (!dateStr) return '';
        try {
            return formatDistanceToNow(new Date(dateStr), { addSuffix: true });
        } catch {
            return '';
        }
    };

    return (
        <>
            {/* Mobile menu button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-dark-800 rounded-lg border border-dark-700"
            >
                {isOpen ? <X size={20} /> : <Menu size={20} />}
            </button>

            {/* Backdrop */}
            {isOpen && (
                <div
                    className="lg:hidden fixed inset-0 bg-black/50 z-30"
                    onClick={() => setIsOpen(false)}
                />
            )}

            {/* Sidebar */}
            <aside
                className={clsx(
                    'fixed lg:static inset-y-0 left-0 z-40 w-72 bg-dark-900 border-r border-dark-800 flex flex-col transform transition-transform lg:transform-none',
                    isOpen ? 'translate-x-0' : '-translate-x-full'
                )}
            >
                {/* Header */}
                <div className="p-4 border-b border-dark-800">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                            <MessageSquare size={20} />
                        </div>
                        <div>
                            <h1 className="font-semibold">RAG Chatbot</h1>
                            <div className="flex items-center gap-2 text-xs text-dark-400">
                                <span
                                    className={clsx(
                                        'w-2 h-2 rounded-full',
                                        status?.status === 'online'
                                            ? 'bg-green-500 animate-pulse-slow'
                                            : 'bg-red-500'
                                    )}
                                />
                                {status?.model || 'Connecting...'}
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={onNewSession}
                        disabled={isLoading}
                        className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 rounded-lg font-medium transition-colors"
                    >
                        <Plus size={18} />
                        New Chat
                    </button>
                </div>

                {/* Sessions list */}
                <div className="flex-1 overflow-y-auto p-2">
                    {isLoading && sessions.length === 0 ? (
                        <div className="flex items-center justify-center py-8">
                            <Loader2 className="animate-spin text-dark-500" size={24} />
                        </div>
                    ) : sessions.length === 0 ? (
                        <div className="text-center py-8 text-dark-500 text-sm">
                            No conversations yet
                        </div>
                    ) : (
                        <div className="space-y-1">
                            {sessions.map((session) => (
                                <div
                                    key={session.session_id}
                                    onClick={() => {
                                        onSelectSession(session.session_id);
                                        setIsOpen(false);
                                    }}
                                    className={clsx(
                                        'group flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-colors',
                                        activeSessionId === session.session_id
                                            ? 'bg-dark-800'
                                            : 'hover:bg-dark-800/50'
                                    )}
                                >
                                    <MessageSquare
                                        size={18}
                                        className="mt-0.5 text-dark-500 flex-shrink-0"
                                    />
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm truncate">
                                            {session.preview || 'New conversation'}
                                        </p>
                                        <p className="text-xs text-dark-500 mt-0.5">
                                            {formatTime(session.last_message_at)} ·{' '}
                                            {session.message_count} messages
                                        </p>
                                    </div>
                                    <button
                                        onClick={(e) => handleDelete(e, session.session_id)}
                                        disabled={deletingId === session.session_id}
                                        className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-dark-700 rounded transition-all"
                                    >
                                        {deletingId === session.session_id ? (
                                            <Loader2 size={14} className="animate-spin" />
                                        ) : (
                                            <Trash2 size={14} className="text-dark-400" />
                                        )}
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-dark-800">
                    <div className="flex items-center gap-2 text-xs text-dark-500">
                        <Settings size={14} />
                        <span>
                            {status?.vector_store} · {status?.collection}
                        </span>
                    </div>
                </div>
            </aside>
        </>
    );
}