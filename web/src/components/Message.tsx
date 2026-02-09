'use client';

import { useState, useMemo, memo } from 'react';
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

const CitationBadge = memo(function CitationBadge({ citation }: { citation: Citation }) {
    const [expanded, setExpanded] = useState(false);

    const icon = useMemo(() => {
        switch (citation.source_type) {
            case 'wikipedia':
                return <BookOpen size={12} />;
            case 'pdf':
                return <FileText size={12} />;
            default:
                return <FileText size={12} />;
        }
    }, [citation.source_type]);

    return (
        <div className="border border-dark-700 rounded-lg overflow-hidden">
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center gap-2 px-3 py-2 hover:bg-dark-700/50 transition-colors text-left"
            >
                <span className="flex items-center justify-center w-5 h-5 rounded bg-primary-600/20 text-primary-400 text-xs font-medium">
                    {citation.number}
                </span>
                <span className="text-dark-400">{icon}</span>
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
});

const CitationsPanel = memo(function CitationsPanel({ citations }: { citations: Citation[] }) {
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
});

// Parse inline markdown (bold, italic, code, links, citations)
// Defined outside component to avoid recreation
// Uses safe regex patterns to prevent catastrophic backtracking
function parseInline(text: string, keyPrefix: string = ''): (string | JSX.Element)[] {
    const elements: (string | JSX.Element)[] = [];
    let remaining = text;
    let key = 0;
    let iterations = 0;
    const MAX_ITERATIONS = 10000; // Safety limit to prevent infinite loops

    while (remaining.length > 0 && iterations < MAX_ITERATIONS) {
        iterations++;

        // Citations [1], [2], etc.
        const citationMatch = remaining.match(/^\[(\d+)\]/);
        if (citationMatch) {
            elements.push(
                <span
                    key={`${keyPrefix}-${key++}`}
                    className="inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 text-xs font-medium bg-primary-600/20 text-primary-400 rounded"
                >
                    {citationMatch[0]}
                </span>
            );
            remaining = remaining.substring(citationMatch[0].length);
            continue;
        }

        // Links [text](url) - use negated character classes to prevent backtracking
        const linkMatch = remaining.match(/^\[([^\]]{1,500})\]\(([^)]{1,1000})\)/);
        if (linkMatch) {
            elements.push(
                <a
                    key={`${keyPrefix}-${key++}`}
                    href={linkMatch[2]}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary-400 hover:text-primary-300 underline"
                >
                    {linkMatch[1]}
                </a>
            );
            remaining = remaining.substring(linkMatch[0].length);
            continue;
        }

        // Inline code `code` - limit length to prevent issues
        const codeMatch = remaining.match(/^`([^`]{1,1000})`/);
        if (codeMatch) {
            elements.push(
                <code key={`${keyPrefix}-${key++}`} className="px-1.5 py-0.5 bg-dark-700 rounded text-xs font-mono">
                    {codeMatch[1]}
                </code>
            );
            remaining = remaining.substring(codeMatch[0].length);
            continue;
        }

        // Bold **text** - use specific non-backtracking pattern
        const boldStarMatch = remaining.match(/^\*\*([^*]+)\*\*/);
        if (boldStarMatch) {
            elements.push(
                <strong key={`${keyPrefix}-${key++}`} className="font-semibold">
                    {boldStarMatch[1]}
                </strong>
            );
            remaining = remaining.substring(boldStarMatch[0].length);
            continue;
        }

        // Bold __text__
        const boldUnderMatch = remaining.match(/^__([^_]+)__/);
        if (boldUnderMatch) {
            elements.push(
                <strong key={`${keyPrefix}-${key++}`} className="font-semibold">
                    {boldUnderMatch[1]}
                </strong>
            );
            remaining = remaining.substring(boldUnderMatch[0].length);
            continue;
        }

        // Italic *text* - use specific non-backtracking pattern (not preceded by *)
        const italicStarMatch = remaining.match(/^\*([^*]+)\*/);
        if (italicStarMatch) {
            elements.push(
                <em key={`${keyPrefix}-${key++}`} className="italic">
                    {italicStarMatch[1]}
                </em>
            );
            remaining = remaining.substring(italicStarMatch[0].length);
            continue;
        }

        // Italic _text_
        const italicUnderMatch = remaining.match(/^_([^_]+)_/);
        if (italicUnderMatch) {
            elements.push(
                <em key={`${keyPrefix}-${key++}`} className="italic">
                    {italicUnderMatch[1]}
                </em>
            );
            remaining = remaining.substring(italicUnderMatch[0].length);
            continue;
        }

        // Regular text - find next special character
        const nextSpecial = remaining.search(/[\[\*_`]/);
        if (nextSpecial === -1) {
            elements.push(remaining);
            break;
        }
        if (nextSpecial === 0) {
            // Special char at start but no pattern matched - treat as regular text
            elements.push(remaining[0]);
            remaining = remaining.substring(1);
        } else {
            elements.push(remaining.substring(0, nextSpecial));
            remaining = remaining.substring(nextSpecial);
        }
    }

    // If we hit the iteration limit, just return remaining text as-is
    if (iterations >= MAX_ITERATIONS && remaining.length > 0) {
        elements.push(remaining);
    }

    return elements;
}

// Custom markdown renderer with citation support
// Defined outside component to avoid recreation
function renderMarkdownContent(content: string): JSX.Element {
    const lines = content.split('\n');
    const elements: JSX.Element[] = [];
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];

        // Code blocks
        if (line.trim().startsWith('```')) {
            const codeLines: string[] = [];
            i++;
            while (i < lines.length && !lines[i].trim().startsWith('```')) {
                codeLines.push(lines[i]);
                i++;
            }
            elements.push(
                <pre key={`code-${i}`} className="bg-dark-700 rounded p-3 my-2 overflow-x-auto">
                    <code className="text-xs font-mono">{codeLines.join('\n')}</code>
                </pre>
            );
            i++;
            continue;
        }

        // Headers
        if (line.startsWith('### ')) {
            elements.push(
                <h3 key={`h3-${i}`} className="text-base font-bold mt-2 mb-1">
                    {parseInline(line.substring(4), `h3-${i}`)}
                </h3>
            );
            i++;
            continue;
        }
        if (line.startsWith('## ')) {
            elements.push(
                <h2 key={`h2-${i}`} className="text-lg font-bold mt-3 mb-2">
                    {parseInline(line.substring(3), `h2-${i}`)}
                </h2>
            );
            i++;
            continue;
        }
        if (line.startsWith('# ')) {
            elements.push(
                <h1 key={`h1-${i}`} className="text-xl font-bold mt-4 mb-2">
                    {parseInline(line.substring(2), `h1-${i}`)}
                </h1>
            );
            i++;
            continue;
        }

        // Blockquotes
        if (line.startsWith('> ')) {
            elements.push(
                <blockquote key={`quote-${i}`} className="border-l-2 border-dark-600 pl-3 my-2 italic">
                    {parseInline(line.substring(2), `quote-${i}`)}
                </blockquote>
            );
            i++;
            continue;
        }

        // Unordered lists
        if (line.match(/^[\s]*[-*+]\s/)) {
            const listItems: JSX.Element[] = [];
            const listStart = i;
            while (i < lines.length && lines[i].match(/^[\s]*[-*+]\s/)) {
                const match = lines[i].match(/^[\s]*[-*+]\s(.+)$/);
                if (match) {
                    listItems.push(<li key={`li-${i}`}>{parseInline(match[1], `li-${i}`)}</li>);
                }
                i++;
            }
            elements.push(
                <ul key={`ul-${listStart}`} className="list-disc list-inside my-2 space-y-1">
                    {listItems}
                </ul>
            );
            continue;
        }

        // Ordered lists
        if (line.match(/^[\s]*\d+\.\s/)) {
            const listItems: JSX.Element[] = [];
            const listStart = i;
            while (i < lines.length && lines[i].match(/^[\s]*\d+\.\s/)) {
                const match = lines[i].match(/^[\s]*\d+\.\s(.+)$/);
                if (match) {
                    listItems.push(<li key={`oli-${i}`}>{parseInline(match[1], `oli-${i}`)}</li>);
                }
                i++;
            }
            elements.push(
                <ol key={`ol-${listStart}`} className="list-decimal list-inside my-2 space-y-1">
                    {listItems}
                </ol>
            );
            continue;
        }

        // Empty lines
        if (line.trim() === '') {
            i++;
            continue;
        }

        // Regular paragraphs
        elements.push(
            <p key={`p-${i}`} className="mb-2 last:mb-0">
                {parseInline(line, `p-${i}`)}
            </p>
        );
        i++;
    }

    return <div>{elements}</div>;
}

function MessageComponent({ message, showTimestamp = true }: MessageProps) {
    const [copied, setCopied] = useState(false);
    const isUser = message.role === 'user';

    const handleCopy = async () => {
        await navigator.clipboard.writeText(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const formattedTime = useMemo(() => {
        if (!message.timestamp) return '';
        try {
            return format(new Date(message.timestamp), 'HH:mm');
        } catch {
            return '';
        }
    }, [message.timestamp]);

    // Memoize the expensive markdown rendering
    const renderedContent = useMemo(() => {
        if (isUser) {
            return <span className="whitespace-pre-wrap">{message.content}</span>;
        }
        return renderMarkdownContent(message.content);
    }, [message.content, isUser]);

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
                    <div className="text-sm leading-relaxed">
                        {renderedContent}
                    </div>
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
                    {showTimestamp && formattedTime && (
                        <span>{formattedTime}</span>
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

// Memoize the entire Message component to prevent unnecessary re-renders
const Message = memo(MessageComponent, (prevProps, nextProps) => {
    // Only re-render if the message content or timestamp changed
    return (
        prevProps.message.content === nextProps.message.content &&
        prevProps.message.timestamp === nextProps.message.timestamp &&
        prevProps.message.role === nextProps.message.role &&
        prevProps.showTimestamp === nextProps.showTimestamp
    );
});

export default Message;