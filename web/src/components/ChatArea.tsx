'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Loader2, Bot, Sparkles, Lightbulb } from 'lucide-react';
import clsx from 'clsx';
import Message from './Message';
import StrategyBadge from './StrategyBadge';
import type { Message as MessageType } from '@/types';

interface ChatAreaProps {
    messages: MessageType[];
    isLoading: boolean;
    isLoadingMore?: boolean;
    hasMoreMessages?: boolean;
    onSendMessage: (message: string) => void;
    onLoadMore?: () => void;
    onOpenSettings?: () => void;
}

const SUGGESTIONS = [
    { text: 'What is artificial intelligence?', icon: '🤖' },
    { text: 'Explain how neural networks work', icon: '🧠' },
    { text: 'What is deep learning?', icon: '📊' },
    { text: 'How does NLP work?', icon: '💬' },
];

function TypingIndicator() {
    return (
        <div className="flex gap-3 animate-fade-in">
            <div className="w-8 h-8 rounded-lg bg-dark-700 flex items-center justify-center flex-shrink-0">
                <Bot size={16} />
            </div>
            <div className="bg-dark-800 rounded-2xl rounded-tl-sm px-4 py-3">
                <div className="flex gap-1.5">
                    <span className="w-2 h-2 bg-dark-500 rounded-full typing-dot" />
                    <span className="w-2 h-2 bg-dark-500 rounded-full typing-dot" />
                    <span className="w-2 h-2 bg-dark-500 rounded-full typing-dot" />
                </div>
            </div>
        </div>
    );
}

function WelcomeScreen({
    onSuggestionClick,
}: {
    onSuggestionClick: (text: string) => void;
}) {
    return (
        <div className="h-full flex flex-col items-center justify-center p-8">
            <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center mb-6 shadow-lg shadow-primary-500/20">
                <Sparkles size={40} />
            </div>

            <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-white to-dark-300 bg-clip-text text-transparent">
                RAG Chatbot
            </h2>

            <p className="text-dark-400 text-center max-w-md mb-8">
                Ask me anything about AI, machine learning, neural networks, and more.
                I'll search my knowledge base and cite my sources.
            </p>

            <div className="w-full max-w-lg">
                <div className="flex items-center gap-2 text-sm text-dark-500 mb-3">
                    <Lightbulb size={16} />
                    <span>Try asking:</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {SUGGESTIONS.map((suggestion) => (
                        <button
                            key={suggestion.text}
                            onClick={() => onSuggestionClick(suggestion.text)}
                            className="flex items-center gap-3 px-4 py-3 bg-dark-800/50 hover:bg-dark-800 border border-dark-700 hover:border-dark-600 rounded-xl text-sm text-left transition-all group"
                        >
                            <span className="text-lg group-hover:scale-110 transition-transform">
                                {suggestion.icon}
                            </span>
                            <span className="text-dark-300 group-hover:text-white transition-colors">
                                {suggestion.text}
                            </span>
                        </button>
                    ))}
                </div>
            </div>

            <div className="mt-8 flex items-center gap-2 text-xs text-dark-600">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span>Powered by RAG with citation tracking</span>
            </div>
        </div>
    );
}

export default function ChatArea({
    messages,
    isLoading,
    isLoadingMore = false,
    hasMoreMessages = false,
    onSendMessage,
    onLoadMore,
    onOpenSettings,
}: ChatAreaProps) {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const messagesContainerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const prevMessageCountRef = useRef(0);
    const prevScrollHeightRef = useRef(0);
    const isLoadingMoreRef = useRef(false);
    const lastScrollCheckRef = useRef(0); // For throttling scroll handler

    const scrollToBottom = useCallback(() => {
        // Use requestAnimationFrame for smoother scrolling
        requestAnimationFrame(() => {
            messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        });
    }, []);

    // Reset ref when isLoadingMore prop changes to false (handles both success and error cases)
    useEffect(() => {
        if (!isLoadingMore) {
            isLoadingMoreRef.current = false;
        }
    }, [isLoadingMore]);

    // Maintain scroll position when loading older messages
    useEffect(() => {
        if (isLoadingMoreRef.current && messagesContainerRef.current && !isLoadingMore) {
            const container = messagesContainerRef.current;
            const newScrollHeight = container.scrollHeight;
            const scrollDiff = newScrollHeight - prevScrollHeightRef.current;
            container.scrollTop = scrollDiff;
        }
    }, [messages, isLoadingMore]);

    // Only scroll to bottom when new messages are added at the end
    useEffect(() => {
        const messageCount = messages.length;
        const wasAtBottom = prevMessageCountRef.current < messageCount;

        if (wasAtBottom && !isLoadingMoreRef.current) {
            prevMessageCountRef.current = messageCount;
            scrollToBottom();
        }
    }, [messages.length, scrollToBottom]);

    // Scroll to bottom on initial load or when loading state ends
    useEffect(() => {
        if (!isLoading && messages.length > 0 && prevMessageCountRef.current === 0) {
            prevMessageCountRef.current = messages.length;
            scrollToBottom();
        }
    }, [isLoading, messages.length, scrollToBottom]);

    // Handle scroll to detect when user scrolls to top (throttled)
    const handleScroll = useCallback(() => {
        // Throttle: only check every 100ms
        const now = Date.now();
        if (now - lastScrollCheckRef.current < 100) {
            return;
        }
        lastScrollCheckRef.current = now;

        // Use ref as primary guard to prevent multiple rapid calls
        // Also skip during initial loading to prevent triggering on mount
        if (
            !messagesContainerRef.current ||
            !hasMoreMessages ||
            isLoading ||
            isLoadingMore ||
            isLoadingMoreRef.current ||
            !onLoadMore ||
            messages.length === 0
        ) {
            return;
        }

        const container = messagesContainerRef.current;
        // Load more when scrolled within 100px of the top
        // Also check that container has enough content to scroll (prevents triggering on small conversations)
        if (container.scrollTop < 100 && container.scrollHeight > container.clientHeight) {
            isLoadingMoreRef.current = true;
            prevScrollHeightRef.current = container.scrollHeight;
            onLoadMore();
        }
    }, [hasMoreMessages, isLoading, isLoadingMore, onLoadMore, messages.length]);

    // Focus input on mount
    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;
        onSendMessage(input.trim());
        setInput('');
    };

    const handleSuggestion = (suggestion: string) => {
        onSendMessage(suggestion);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <div className="flex-1 flex flex-col h-full">
            {/* Header with strategy badge - shown when there are messages */}
            {messages.length > 0 && (
                <div className="flex items-center justify-between px-4 py-2 border-b border-dark-800 bg-dark-900/50 backdrop-blur-sm">
                    <div className="flex items-center gap-2 text-sm text-dark-400">
                        <Bot size={16} />
                        <span>Chat</span>
                        <span className="text-dark-600">•</span>
                        <span>{messages.length} messages</span>
                    </div>
                    <StrategyBadge onClick={onOpenSettings} />
                </div>
            )}

            {/* Messages area */}
            <div
                ref={messagesContainerRef}
                className="flex-1 overflow-y-auto"
                onScroll={handleScroll}
            >
                {messages.length === 0 ? (
                    <WelcomeScreen onSuggestionClick={handleSuggestion} />
                ) : (
                    <div className="max-w-3xl mx-auto p-4 space-y-6">
                        {/* Loading indicator for older messages */}
                        {isLoadingMore && (
                            <div className="flex items-center justify-center py-4">
                                <Loader2 className="animate-spin text-dark-500" size={20} />
                                <span className="ml-2 text-sm text-dark-500">Loading older messages...</span>
                            </div>
                        )}

                        {/* Load more indicator when there are more messages */}
                        {hasMoreMessages && !isLoadingMore && (
                            <div className="flex items-center justify-center py-2">
                                <span className="text-xs text-dark-600">Scroll up to load more</span>
                            </div>
                        )}

                        {messages.map((message, index) => (
                            <Message
                                key={message.id ?? message.timestamp ?? `msg-${index}`}
                                message={message}
                                showTimestamp={true}
                            />
                        ))}

                        {isLoading && <TypingIndicator />}

                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input area */}
            <div className="border-t border-dark-800 p-4 bg-dark-950/80 backdrop-blur-sm">
                <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
                    <div className="flex gap-3">
                        <div className="flex-1 relative">
                            <input
                                ref={inputRef}
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Ask anything..."
                                disabled={isLoading}
                                className={clsx(
                                    'w-full px-4 py-3 pr-12 bg-dark-800 border border-dark-700 rounded-xl',
                                    'focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500',
                                    'disabled:opacity-50 transition-all',
                                    'placeholder:text-dark-500'
                                )}
                            />
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-dark-600">
                                {input.length > 0 && `${input.length}`}
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className={clsx(
                                'px-4 py-3 rounded-xl font-medium transition-all',
                                'bg-primary-600 hover:bg-primary-500',
                                'disabled:opacity-50 disabled:hover:bg-primary-600 disabled:cursor-not-allowed',
                                'flex items-center gap-2'
                            )}
                        >
                            {isLoading ? (
                                <Loader2 size={20} className="animate-spin" />
                            ) : (
                                <Send size={20} />
                            )}
                        </button>
                    </div>

                    <p className="text-center text-xs text-dark-600 mt-3">
                        Responses include citations from the knowledge base • Press Enter to send
                    </p>
                </form>
            </div>
        </div>
    );
}