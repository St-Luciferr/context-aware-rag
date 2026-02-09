'use client';

import { useState, useEffect } from 'react';
import {
    X,
    Plus,
    Loader2,
    RefreshCw,
    BookOpen,
    Check,
    AlertCircle,
    RotateCcw,
    Database,
    Clock,
    Lock,
} from 'lucide-react';
import clsx from 'clsx';
import { api } from '@/lib/api';
import type { TopicInfo } from '@/types';

interface TopicsManagerProps {
    onIngestComplete?: () => void;
}

export default function TopicsManager({ onIngestComplete }: TopicsManagerProps) {
    const [topics, setTopics] = useState<TopicInfo[]>([]);
    const [pendingCount, setPendingCount] = useState(0);
    const [newTopic, setNewTopic] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [isAdding, setIsAdding] = useState(false);
    const [isIngesting, setIsIngesting] = useState(false);
    const [isFullIngesting, setIsFullIngesting] = useState(false);
    const [removingTopic, setRemovingTopic] = useState<string | null>(null);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    useEffect(() => {
        fetchTopics();
    }, []);

    const fetchTopics = async () => {
        setIsLoading(true);
        try {
            const data = await api.getTopics();
            setTopics(data.topics);
            setPendingCount(data.pending_count);
        } catch (err) {
            console.error('Failed to fetch topics:', err);
            setMessage({ type: 'error', text: 'Failed to load topics' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleAddTopic = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newTopic.trim() || isAdding) return;

        setIsAdding(true);
        setMessage(null);

        try {
            const result = await api.addTopic({ topic: newTopic.trim() });
            if (result.success) {
                setNewTopic('');
                setMessage({ type: 'success', text: result.message });
                await fetchTopics(); // Refresh the list
            }
        } catch (err: unknown) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to add topic';
            setMessage({ type: 'error', text: errorMessage });
        } finally {
            setIsAdding(false);
        }
    };

    const handleRemoveTopic = async (topic: string) => {
        if (removingTopic) return;

        setRemovingTopic(topic);
        setMessage(null);

        try {
            const result = await api.removeTopic(topic);
            if (result.success) {
                setMessage({ type: 'success', text: result.message });
                await fetchTopics();
            }
        } catch (err: unknown) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to remove topic';
            setMessage({ type: 'error', text: errorMessage });
        } finally {
            setRemovingTopic(null);
        }
    };

    const handleResetTopics = async () => {
        setMessage(null);
        try {
            const result = await api.resetTopics();
            if (result.success) {
                setMessage({ type: 'success', text: result.message });
                await fetchTopics();
            }
        } catch (err: unknown) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to reset topics';
            setMessage({ type: 'error', text: errorMessage });
        }
    };

    const handleIngestPending = async () => {
        if (isIngesting || isFullIngesting) return;

        setIsIngesting(true);
        setMessage(null);

        try {
            const result = await api.ingestPendingTopics();
            if (result.status === 'success') {
                setMessage({
                    type: 'success',
                    text: `${result.message}`
                });
                await fetchTopics();
                onIngestComplete?.();
            } else if (result.status === 'skipped') {
                setMessage({ type: 'success', text: result.message });
            } else {
                setMessage({ type: 'error', text: result.message });
            }
        } catch (err: unknown) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to ingest topics';
            setMessage({ type: 'error', text: errorMessage });
        } finally {
            setIsIngesting(false);
        }
    };

    const handleFullReingest = async () => {
        if (isIngesting || isFullIngesting) return;

        if (!confirm('This will delete all existing data and re-ingest everything. Continue?')) {
            return;
        }

        setIsFullIngesting(true);
        setMessage(null);

        try {
            const result = await api.fullReingest();
            if (result.status === 'success') {
                setMessage({
                    type: 'success',
                    text: `Full re-ingestion complete. ${result.document_count} chunks indexed.`
                });
                await fetchTopics();
                onIngestComplete?.();
            } else {
                setMessage({ type: 'error', text: result.message });
            }
        } catch (err: unknown) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to re-ingest';
            setMessage({ type: 'error', text: errorMessage });
        } finally {
            setIsFullIngesting(false);
        }
    };

    const ingestedTopics = topics.filter(t => t.is_ingested);
    const pendingTopics = topics.filter(t => !t.is_ingested);

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-2">
                <BookOpen size={16} className="text-primary-400" />
                <h3 className="font-medium">Knowledge Base Topics</h3>
            </div>
            <p className="text-sm text-dark-400">
                Add Wikipedia topics to expand the knowledge base. Default topics (from config) cannot be removed.
            </p>

            {/* Topics List */}
            {isLoading ? (
                <div className="flex items-center justify-center py-6">
                    <Loader2 className="animate-spin text-dark-500" size={24} />
                </div>
            ) : (
                <div className="space-y-4">
                    {/* Ingested Topics */}
                    <div>
                        <div className="flex items-center gap-2 mb-2">
                            <Database size={14} className="text-green-400" />
                            <span className="text-sm font-medium text-dark-300">
                                In Knowledge Base ({ingestedTopics.length})
                            </span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {ingestedTopics.map((topic) => (
                                <div
                                    key={topic.name}
                                    className={clsx(
                                        'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm',
                                        'bg-green-500/10 border border-green-500/30',
                                        removingTopic === topic.name && 'opacity-50'
                                    )}
                                >
                                    <Check size={12} className="text-green-400" />
                                    <span>{topic.name}</span>
                                    {topic.is_default ? (
                                        <span title="Default topic (from config)">
                                            <Lock size={12} className="text-dark-500 ml-1" />
                                        </span>
                                    ) : (
                                        <button
                                            onClick={() => handleRemoveTopic(topic.name)}
                                            disabled={removingTopic !== null}
                                            className="p-0.5 rounded hover:bg-dark-700 transition-colors ml-1"
                                            title={`Remove ${topic.name}`}
                                        >
                                            {removingTopic === topic.name ? (
                                                <Loader2 size={12} className="animate-spin" />
                                            ) : (
                                                <X size={12} className="text-dark-400 hover:text-red-400" />
                                            )}
                                        </button>
                                    )}
                                </div>
                            ))}
                            {ingestedTopics.length === 0 && (
                                <span className="text-sm text-dark-500 italic">No topics ingested yet</span>
                            )}
                        </div>
                    </div>

                    {/* Pending Topics */}
                    {pendingTopics.length > 0 && (
                        <div>
                            <div className="flex items-center gap-2 mb-2">
                                <Clock size={14} className="text-amber-400" />
                                <span className="text-sm font-medium text-dark-300">
                                    Pending Ingestion ({pendingTopics.length})
                                </span>
                            </div>
                            <div className="flex flex-wrap gap-2">
                                {pendingTopics.map((topic) => (
                                    <div
                                        key={topic.name}
                                        className={clsx(
                                            'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm',
                                            'bg-amber-500/10 border border-amber-500/30',
                                            removingTopic === topic.name && 'opacity-50'
                                        )}
                                    >
                                        <Clock size={12} className="text-amber-400" />
                                        <span>{topic.name}</span>
                                        {topic.is_default ? (
                                            <span title="Default topic (from config)">
                                                <Lock size={12} className="text-dark-500 ml-1" />
                                            </span>
                                        ) : (
                                            <button
                                                onClick={() => handleRemoveTopic(topic.name)}
                                                disabled={removingTopic !== null}
                                                className="p-0.5 rounded hover:bg-dark-700 transition-colors ml-1"
                                                title={`Remove ${topic.name}`}
                                            >
                                                {removingTopic === topic.name ? (
                                                    <Loader2 size={12} className="animate-spin" />
                                                ) : (
                                                    <X size={12} className="text-dark-400 hover:text-red-400" />
                                                )}
                                            </button>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Add Topic Form */}
                    <form onSubmit={handleAddTopic} className="flex gap-2 mt-3">
                        <input
                            type="text"
                            value={newTopic}
                            onChange={(e) => setNewTopic(e.target.value)}
                            placeholder="Enter Wikipedia topic (e.g., 'Transformer model')..."
                            className="flex-1 px-3 py-2 text-sm bg-dark-800 border border-dark-700 rounded-lg focus:outline-none focus:border-primary-500 placeholder:text-dark-500"
                        />
                        <button
                            type="submit"
                            disabled={!newTopic.trim() || isAdding}
                            className={clsx(
                                'flex items-center gap-1.5 px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                                newTopic.trim() && !isAdding
                                    ? 'bg-primary-600 hover:bg-primary-700 text-white'
                                    : 'bg-dark-800 text-dark-500 cursor-not-allowed'
                            )}
                        >
                            {isAdding ? (
                                <Loader2 size={16} className="animate-spin" />
                            ) : (
                                <Plus size={16} />
                            )}
                            Add
                        </button>
                    </form>
                </div>
            )}

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-2 pt-2">
                <button
                    onClick={handleResetTopics}
                    disabled={isIngesting || isFullIngesting}
                    className="flex items-center gap-1.5 px-3 py-2 text-sm text-dark-400 hover:text-white hover:bg-dark-800 rounded-lg transition-colors"
                >
                    <RotateCcw size={14} />
                    Clear Additional
                </button>
                <button
                    onClick={handleFullReingest}
                    disabled={isIngesting || isFullIngesting}
                    className="flex items-center gap-1.5 px-3 py-2 text-sm text-dark-400 hover:text-white hover:bg-dark-800 rounded-lg transition-colors"
                >
                    {isFullIngesting ? (
                        <Loader2 size={14} className="animate-spin" />
                    ) : (
                        <RefreshCw size={14} />
                    )}
                    Full Re-ingest
                </button>
                <button
                    onClick={handleIngestPending}
                    disabled={isIngesting || isFullIngesting || pendingCount === 0}
                    className={clsx(
                        'flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-lg transition-colors ml-auto',
                        isIngesting
                            ? 'bg-amber-600/20 text-amber-400 border border-amber-600/30'
                            : pendingCount > 0
                                ? 'bg-green-600 hover:bg-green-700 text-white'
                                : 'bg-dark-800 text-dark-500 cursor-not-allowed'
                    )}
                >
                    {isIngesting ? (
                        <>
                            <Loader2 size={16} className="animate-spin" />
                            Ingesting...
                        </>
                    ) : (
                        <>
                            <Database size={16} />
                            Ingest New Topics {pendingCount > 0 && `(${pendingCount})`}
                        </>
                    )}
                </button>
            </div>

            {/* Message */}
            {message && (
                <div
                    className={clsx(
                        'flex items-center gap-2 p-3 rounded-lg text-sm animate-fade-in',
                        message.type === 'success'
                            ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                            : 'bg-red-500/10 text-red-400 border border-red-500/20'
                    )}
                >
                    {message.type === 'success' ? <Check size={16} /> : <AlertCircle size={16} />}
                    {message.text}
                </div>
            )}

            {/* Info box */}
            <div className="p-3 bg-dark-800/50 rounded-lg border border-dark-700">
                <div className="flex items-start gap-2">
                    <AlertCircle size={16} className="text-dark-500 mt-0.5 flex-shrink-0" />
                    <p className="text-xs text-dark-500">
                        <strong>How it works:</strong> Add topics, then click &quot;Ingest New Topics&quot; to add them
                        to the knowledge base incrementally. Use &quot;Full Re-ingest&quot; to rebuild everything from scratch.
                    </p>
                </div>
            </div>
        </div>
    );
}
