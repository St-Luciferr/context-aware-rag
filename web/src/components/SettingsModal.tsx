'use client';

import { useState, useEffect } from 'react';
import {
    X,
    Settings,
    Loader2,
    Check,
    Layers,
    Clock,
    FileText,
    Sparkles,
    Info,
} from 'lucide-react';
import clsx from 'clsx';
import { api } from '@/lib/api';
import type { StrategyInfo } from '@/types';

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
}

const STRATEGY_ICONS: Record<string, React.ReactNode> = {
    sliding_window: <Layers size={20} />,
    token_budget: <Clock size={20} />,
    summarization: <FileText size={20} />,
};

const STRATEGY_COLORS: Record<string, string> = {
    sliding_window: 'from-blue-500 to-blue-600',
    token_budget: 'from-green-500 to-green-600',
    summarization: 'from-amber-500 to-amber-600',
};

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
    const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
    const [currentStrategy, setCurrentStrategy] = useState<string>('');
    const [isLoading, setIsLoading] = useState(true);
    const [isChanging, setIsChanging] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    useEffect(() => {
        if (isOpen) {
            fetchStrategies();
        }
    }, [isOpen]);

    const fetchStrategies = async () => {
        setIsLoading(true);
        try {
            const data = await api.getStrategies();
            setStrategies(data.strategies);
            setCurrentStrategy(data.current);
        } catch (err) {
            console.error('Failed to fetch strategies:', err);
            setMessage({ type: 'error', text: 'Failed to load strategies' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleChangeStrategy = async (strategyId: string) => {
        if (strategyId === currentStrategy || isChanging) return;

        setIsChanging(true);
        setMessage(null);

        try {
            const result = await api.changeStrategy({ strategy: strategyId });
            if (result.success) {
                setCurrentStrategy(result.current_strategy);
                setMessage({ type: 'success', text: result.message });
            }
        } catch (err) {
            console.error('Failed to change strategy:', err);
            setMessage({ type: 'error', text: 'Failed to change strategy' });
        } finally {
            setIsChanging(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative w-full max-w-lg bg-dark-900 border border-dark-700 rounded-2xl shadow-2xl animate-fade-in">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-dark-700">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                            <Settings size={20} />
                        </div>
                        <div>
                            <h2 className="font-semibold text-lg">Settings</h2>
                            <p className="text-xs text-dark-400">Configure chat behavior</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-dark-800 rounded-lg transition-colors"
                    >
                        <X size={20} className="text-dark-400" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-4">
                    {/* Section: History Strategy */}
                    <div className="mb-4">
                        <div className="flex items-center gap-2 mb-3">
                            <Sparkles size={16} className="text-primary-400" />
                            <h3 className="font-medium">History Management Strategy</h3>
                        </div>
                        <p className="text-sm text-dark-400 mb-4">
                            Choose how conversation history is sent to the AI model. This affects response quality and speed.
                        </p>

                        {isLoading ? (
                            <div className="flex items-center justify-center py-8">
                                <Loader2 className="animate-spin text-dark-500" size={24} />
                            </div>
                        ) : (
                            <div className="space-y-2">
                                {strategies.map((strategy) => (
                                    <button
                                        key={strategy.id}
                                        onClick={() => handleChangeStrategy(strategy.id)}
                                        disabled={isChanging}
                                        className={clsx(
                                            'w-full flex items-start gap-3 p-3 rounded-xl border transition-all text-left',
                                            currentStrategy === strategy.id
                                                ? 'border-primary-500 bg-primary-500/10'
                                                : 'border-dark-700 hover:border-dark-600 hover:bg-dark-800/50'
                                        )}
                                    >
                                        {/* Icon */}
                                        <div
                                            className={clsx(
                                                'w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0',
                                                currentStrategy === strategy.id
                                                    ? `bg-gradient-to-br ${STRATEGY_COLORS[strategy.id]} text-white`
                                                    : 'bg-dark-800 text-dark-400'
                                            )}
                                        >
                                            {STRATEGY_ICONS[strategy.id]}
                                        </div>

                                        {/* Content */}
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2">
                                                <span className="font-medium">{strategy.name}</span>
                                                {currentStrategy === strategy.id && (
                                                    <span className="flex items-center gap-1 text-xs text-primary-400 bg-primary-500/20 px-2 py-0.5 rounded-full">
                                                        <Check size={12} />
                                                        Active
                                                    </span>
                                                )}
                                            </div>
                                            <p className="text-sm text-dark-400 mt-0.5">
                                                {strategy.description}
                                            </p>

                                            {/* Settings preview */}
                                            <div className="flex gap-2 mt-2 flex-wrap">
                                                {Object.entries(strategy.settings).map(([key, value]) => (
                                                    <span
                                                        key={key}
                                                        className="text-xs bg-dark-800 text-dark-400 px-2 py-0.5 rounded"
                                                    >
                                                        {key.replace(/_/g, ' ')}: {value}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Loading indicator when changing */}
                                        {isChanging && currentStrategy !== strategy.id && (
                                            <Loader2 size={16} className="animate-spin text-dark-500" />
                                        )}
                                    </button>
                                ))}
                            </div>
                        )}
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
                            {message.type === 'success' ? <Check size={16} /> : <Info size={16} />}
                            {message.text}
                        </div>
                    )}

                    {/* Info box */}
                    <div className="mt-4 p-3 bg-dark-800/50 rounded-lg border border-dark-700">
                        <div className="flex items-start gap-2">
                            <Info size={16} className="text-dark-500 mt-0.5 flex-shrink-0" />
                            <p className="text-xs text-dark-500">
                                <strong>Tip:</strong> For most conversations, Sliding Window works best.
                                Use Token Budget for longer chats, or Summarization to preserve important
                                context from very long conversations.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="flex justify-end gap-2 p-4 border-t border-dark-700">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-medium text-dark-300 hover:text-white hover:bg-dark-800 rounded-lg transition-colors"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}