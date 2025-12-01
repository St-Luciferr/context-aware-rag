'use client';

import { useState } from 'react';
import { format } from 'date-fns';
import {
    Bot,
    User,
    ChevronDown,
    ChevronUp,
    ExternalLink,
    FileText,
    BookOpen,
    Copy,
    Check,
} from 'lucide-react';
import clsx from 'clsx';
import type { Message as MessageType, Citation } from '@/types';

interface MessageProps {
    message: MessageType;
    showTimestamp?: boolean;
}

function CitationBadge({ citation }: { citation: Citation }) {
    const [expanded, setExpanded] = useState(false);

    const getIcon = () => {
        switch (citation.source_type) {
            case 'wikipedia':
                return <BookOpen size={12} />;
            case 'pdf':
                return <FileText size={12} />;
            default:
                return <FileText size={12} />;
        }
    };

    return (
        <div className="border border-dark-700 rounded-lg overflow-hidden">
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center gap-2 px-3 py-2 hover:bg-dark-700/50 transition-colors text-left"
            >
                <span className="flex items-center justify-center w-5 h-5 rounded bg-primary-600/20 text-primary-400 text-xs font-medium">
                    {citation.number}
                </span>
                <span className="text-dark-400">{getIcon()}</span>
                <span className="flex-1 text-sm truncate">{citation.display}</span>
                {expanded ? (
                    <ChevronUp size={14} className="text-dark-500" />
                ) : (
                    <ChevronDown size={14} className="text-dark-500" />
                )}
            </button>

            {expanded && (
                <div className="px-3 py-2 border-t border-dark-700 bg-dark-800/50">
                    <p className="text-xs text-dark-400 mb-2 line-clamp-3">
                        {citation.content_preview}
                    </p>
                    <div className="flex items-center gap-3 text-xs">
                        {citation.url && (
                            <a
                                href={citation.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-1 text-primary-400 hover:text-primary-300"
                            >
                                <ExternalLink size={12} />
                                View source
                            </a>
                        )}
                        {citation.page !== undefined && (
                            <span className="text-dark-500">Page {citation.page}</span>
                        )}
                        {citation.chunk_id !== undefined && (
                            <span className="text-dark-500">Chunk {citation.chunk_id}</span>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

function CitationsPanel({ citations }: { citations: Citation[] }) {
    const [isOpen, setIsOpen] = useState(false);

    if (!citations || citations.length === 0) return null;

    return (
        <div className="mt-3">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 text-xs text-dark-400 hover:text-dark-300 transition-colors"
            >
                <BookOpen size={14} />
                <span>
                    {citations.length} source{citations.length !== 1 ? 's' : ''}
                </span>
                {isOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>

            {isOpen && (
                <div className="mt-2 space-y-2 animate-fade-in">
                    {citations.map((citation) => (
                        <CitationBadge key={citation.number} citation={citation} />
                    ))}
                </div>
            )}
        </div>
    );
}

export default function Message({ message, showTimestamp = true }: MessageProps) {
    const [copied, setCopied] = useState(false);
    const isUser = message.role === 'user';

    const handleCopy = async () => {
        await navigator.clipboard.writeText(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const formatTime = (timestamp?: string) => {
        if (!timestamp) return '';
        try {
            return format(new Date(timestamp), 'HH:mm');
        } catch {
            return '';
        }
    };

    // Parse content to highlight citation references [1], [2], etc.
    const renderContent = (content: string) => {
        const parts = content.split(/(\[\d+\])/g);
        return parts.map((part, index) => {
            if (/^\[\d+\]$/.test(part)) {
                return (
                    <span
                        key={index}
                        className="inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 text-xs font-medium bg-primary-600/20 text-primary-400 rounded"
                    >
                        {part}
                    </span>
                );
            }
            return part;
        });
    };

    return (
        <div
            className={clsx(
                'group flex gap-3 animate-fade-in',
                isUser ? 'flex-row-reverse' : ''
            )}
        >
            {/* Avatar */}
            <div
                className={clsx(
                    'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0',
                    isUser ? 'bg-primary-600' : 'bg-dark-700'
                )}
            >
                {isUser ? <User size={16} /> : <Bot size={16} />}
            </div>

            {/* Message content */}
            <div className={clsx('flex-1 max-w-[80%]', isUser ? 'text-right' : '')}>
                <div
                    className={clsx(
                        'inline-block px-4 py-3 rounded-2xl',
                        isUser
                            ? 'bg-primary-600 rounded-tr-sm'
                            : 'bg-dark-800 rounded-tl-sm'
                    )}
                >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {isUser ? message.content : renderContent(message.content)}
                    </p>
                </div>

                {/* Citations for assistant messages */}
                {!isUser && message.citations && (
                    <CitationsPanel citations={message.citations} />
                )}

                {/* Footer: timestamp and actions */}
                <div
                    className={clsx(
                        'flex items-center gap-2 mt-1 text-xs text-dark-500',
                        isUser ? 'justify-end' : 'justify-start'
                    )}
                >
                    {showTimestamp && message.timestamp && (
                        <span>{formatTime(message.timestamp)}</span>
                    )}

                    {/* Copy button - show on hover */}
                    <button
                        onClick={handleCopy}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:text-dark-300 transition-all"
                        title="Copy message"
                    >
                        {copied ? <Check size={12} /> : <Copy size={12} />}
                    </button>
                </div>
            </div>
        </div>
    );
}