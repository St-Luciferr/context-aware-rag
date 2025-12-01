'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Bot, Sparkles, Lightbulb } from 'lucide-react';
import clsx from 'clsx';
import Message from './Message';
import type { Message as MessageType } from '@/types';

interface ChatAreaProps {
    messages: MessageType[];
    isLoading: boolean;
    onSendMessage: (message: string) => void;
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
    onSendMessage,
}: ChatAreaProps) {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

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
            {/* Messages area */}
            <div className="flex-1 overflow-y-auto">
                {messages.length === 0 ? (
                    <WelcomeScreen onSuggestionClick={handleSuggestion} />
                ) : (
                    <div className="max-w-3xl mx-auto p-4 space-y-6">
                        {messages.map((message, index) => (
                            <Message
                                key={`${message.role}-${index}-${message.timestamp || index}`}
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