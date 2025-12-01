'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Bot, User, Sparkles } from 'lucide-react';
import clsx from 'clsx';
import type { Message } from '@/types';

interface ChatAreaProps {
    messages: Message[];
    isLoading: boolean;
    onSendMessage: (message: string) => void;
}

const SUGGESTIONS = [
    'What is artificial intelligence?',
    'Explain how neural networks work',
    'What is deep learning?',
    'How does machine learning differ from AI?',
];

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

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;
        onSendMessage(input.trim());
        setInput('');
    };

    const handleSuggestion = (suggestion: string) => {
        onSendMessage(suggestion);
    };

    return (
        <div className="flex-1 flex flex-col h-full">
            {/* Messages area */}
            <div className="flex-1 overflow-y-auto">
                {messages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center p-8">
                        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center mb-6">
                            <Sparkles size={32} />
                        </div>
                        <h2 className="text-2xl font-semibold mb-2">
                            Welcome to RAG Chatbot
                        </h2>
                        <p className="text-dark-400 text-center max-w-md mb-8">
                            Ask me anything about AI, machine learning, neural networks, and
                            more. I'll search my knowledge base to give you accurate answers.
                        </p>
                        <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                            {SUGGESTIONS.map((suggestion) => (
                                <button
                                    key={suggestion}
                                    onClick={() => handleSuggestion(suggestion)}
                                    className="px-4 py-2 bg-dark-800 hover:bg-dark-700 border border-dark-700 rounded-full text-sm transition-colors"
                                >
                                    {suggestion}
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div className="max-w-3xl mx-auto p-4 space-y-6">
                        {messages.map((message, index) => (
                            <div
                                key={index}
                                className={clsx(
                                    'flex gap-4 animate-fade-in',
                                    message.role === 'user' ? 'flex-row-reverse' : ''
                                )}
                            >
                                <div
                                    className={clsx(
                                        'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0',
                                        message.role === 'user'
                                            ? 'bg-primary-600'
                                            : 'bg-dark-700'
                                    )}
                                >
                                    {message.role === 'user' ? (
                                        <User size={16} />
                                    ) : (
                                        <Bot size={16} />
                                    )}
                                </div>
                                <div
                                    className={clsx(
                                        'flex-1 max-w-[80%]',
                                        message.role === 'user' ? 'text-right' : ''
                                    )}
                                >
                                    <div
                                        className={clsx(
                                            'inline-block px-4 py-3 rounded-2xl',
                                            message.role === 'user'
                                                ? 'bg-primary-600 rounded-tr-sm'
                                                : 'bg-dark-800 rounded-tl-sm'
                                        )}
                                    >
                                        <p className="text-sm leading-relaxed whitespace-pre-wrap">
                                            {message.content}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        ))}

                        {/* Typing indicator */}
                        {isLoading && (
                            <div className="flex gap-4 animate-fade-in">
                                <div className="w-8 h-8 rounded-lg bg-dark-700 flex items-center justify-center flex-shrink-0">
                                    <Bot size={16} />
                                </div>
                                <div className="bg-dark-800 rounded-2xl rounded-tl-sm px-4 py-3">
                                    <div className="flex gap-1">
                                        <span className="w-2 h-2 bg-dark-500 rounded-full typing-dot" />
                                        <span className="w-2 h-2 bg-dark-500 rounded-full typing-dot" />
                                        <span className="w-2 h-2 bg-dark-500 rounded-full typing-dot" />
                                    </div>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input area */}
            <div className="border-t border-dark-800 p-4">
                <form
                    onSubmit={handleSubmit}
                    className="max-w-3xl mx-auto flex gap-3"
                >
                    <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type your message..."
                        disabled={isLoading}
                        className="flex-1 px-4 py-3 bg-dark-800 border border-dark-700 rounded-xl focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 disabled:opacity-50 transition-colors"
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="px-4 py-3 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:hover:bg-primary-600 rounded-xl transition-colors"
                    >
                        {isLoading ? (
                            <Loader2 size={20} className="animate-spin" />
                        ) : (
                            <Send size={20} />
                        )}
                    </button>
                </form>
                <p className="text-center text-xs text-dark-500 mt-3">
                    RAG Chatbot uses retrieval-augmented generation for accurate responses
                </p>
            </div>
        </div>
    );
}