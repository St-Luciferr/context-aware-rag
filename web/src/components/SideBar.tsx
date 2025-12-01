'use client';

import { useState } from 'react';
import { formatDistanceToNow, format } from 'date-fns';
import {
    MessageSquare,
    Plus,
    Trash2,
    Menu,
    X,
    Settings,
    Loader2,
    Database,
    Cpu,
    Clock,
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
    const [showSettings, setShowSettings] = useState(false);

    const handleDelete = async (e: React.MouseEvent, sessionId: string) => {
        e.stopPropagation();
        setDeletingId(sessionId);
        await onDeleteSession(sessionId);
        setDeletingId(null);
    };

    const formatTime = (dateStr: string | null) => {
        if (!dateStr) return '';
        try {
            const date = new Date(dateStr);
            const now = new Date();
            const diffHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60);

            if (diffHours < 24) {
                return formatDistanceToNow(date, { addSuffix: true });
            }
            return format(date, 'MMM d');
        } catch {
            return '';
        }
    };

    const truncatePreview = (preview: string | null, maxLength: number = 40) => {
        if (!preview) return 'New conversation';
        if (preview.length <= maxLength) return preview;
        return preview.slice(0, maxLength) + '...';
    };

    return (
        <>
            {/* Mobile menu button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="lg:hidden fixed top-4 left-4 z-50 p-2.5 bg-dark-800 hover:bg-dark-700 rounded-lg border border-dark-700 transition-colors"
            >
                {isOpen ? <X size={20} /> : <Menu size={20} />}
            </button>

            {/* Backdrop */}
            {isOpen && (
                <div
                    className="lg:hidden fixed inset-0 bg-black/60 backdrop-blur-sm z-30"
                    onClick={() => setIsOpen(false)}
                />
            )}

            {/* Sidebar */}
            <aside
                className={clsx(
                    'fixed lg:static inset-y-0 left-0 z-40 w-80 bg-dark-900 border-r border-dark-800 flex flex-col transform transition-transform duration-300 lg:transform-none',
                    isOpen ? 'translate-x-0' : '-translate-x-full'
                )}
            >
                {/* Header */}
                <div className="p-4 border-b border-dark-800">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg shadow-primary-500/20">
                            <MessageSquare size={22} />
                        </div>
                        <div className="flex-1 min-w-0">
                            <h1 className="font-semibold text-lg">RAG Chatbot</h1>
                            <div className="flex items-center gap-2 text-xs text-dark-400">
                                <span
                                    className={clsx(
                                        'w-2 h-2 rounded-full',
                                        status?.status === 'online'
                                            ? 'bg-green-500 animate-pulse-slow'
                                            : 'bg-red-500'
                                    )}
                                />
                                <span className="truncate">
                                    {status?.model || 'Connecting...'}
                                </span>
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={() => {
                            onNewSession();
                            setIsOpen(false);
                        }}
                        disabled={isLoading}
                        className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-primary-600 hover:bg-primary-500 disabled:opacity-50 disabled:hover:bg-primary-600 rounded-xl font-medium transition-all shadow-lg shadow-primary-600/20 hover:shadow-primary-500/30"
                    >
                        <Plus size={18} />
                        New Chat
                    </button>
                </div>

                {/* Sessions list */}
                <div className="flex-1 overflow-y-auto p-2">
                    {isLoading && sessions.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-12 text-dark-500">
                            <Loader2 className="animate-spin mb-2" size={24} />
                            <span className="text-sm">Loading sessions...</span>
                        </div>
                    ) : sessions.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-12 text-dark-500">
                            <MessageSquare size={32} className="mb-2 opacity-50" />
                            <span className="text-sm">No conversations yet</span>
                            <span className="text-xs mt-1">Start a new chat above</span>
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
                                        'group flex items-start gap-3 p-3 rounded-xl cursor-pointer transition-all',
                                        activeSessionId === session.session_id
                                            ? 'bg-dark-800 border border-dark-700'
                                            : 'hover:bg-dark-800/50 border border-transparent'
                                    )}
                                >
                                    <div
                                        className={clsx(
                                            'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors',
                                            activeSessionId === session.session_id
                                                ? 'bg-primary-600/20 text-primary-400'
                                                : 'bg-dark-800 text-dark-500'
                                        )}
                                    >
                                        <MessageSquare size={16} />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm font-medium truncate">
                                            {truncatePreview(session.preview)}
                                        </p>
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className="text-xs text-dark-500">
                                                {session.message_count} messages
                                            </span>
                                            <span className="text-dark-700">·</span>
                                            <span className="text-xs text-dark-500">
                                                {formatTime(session.last_message_at)}
                                            </span>
                                        </div>
                                    </div>
                                    <button
                                        onClick={(e) => handleDelete(e, session.session_id)}
                                        disabled={deletingId === session.session_id}
                                        className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-dark-700 rounded-lg transition-all"
                                        title="Delete session"
                                    >
                                        {deletingId === session.session_id ? (
                                            <Loader2 size={14} className="animate-spin" />
                                        ) : (
                                            <Trash2 size={14} className="text-dark-400 hover:text-red-400" />
                                        )}
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-dark-800 space-y-3">
                    {/* Settings panel toggle */}
                    <button
                        onClick={() => setShowSettings(!showSettings)}
                        className="w-full flex items-center justify-between px-3 py-2 text-sm text-dark-400 hover:text-dark-300 hover:bg-dark-800 rounded-lg transition-colors"
                    >
                        <div className="flex items-center gap-2">
                            <Settings size={16} />
                            <span>System Info</span>
                        </div>
                        <span className="text-xs">{showSettings ? '▲' : '▼'}</span>
                    </button>

                    {/* Expandable settings panel */}
                    {showSettings && status && (
                        <div className="space-y-2 p-3 bg-dark-800/50 rounded-lg text-xs animate-fade-in">
                            <div className="flex items-center gap-2 text-dark-400">
                                <Cpu size={14} />
                                <span>Model:</span>
                                <span className="text-dark-300">{status.model}</span>
                            </div>
                            <div className="flex items-center gap-2 text-dark-400">
                                <Database size={14} />
                                <span>Store:</span>
                                <span className="text-dark-300">{status.vector_store}</span>
                            </div>
                            <div className="flex items-center gap-2 text-dark-400">
                                <Clock size={14} />
                                <span>Collection:</span>
                                <span className="text-dark-300">{status.collection}</span>
                            </div>
                        </div>
                    )}
                </div>
            </aside>
        </>
    );
}