'use client';

import { useState, useEffect } from 'react';
import { Layers, Clock, FileText, Settings } from 'lucide-react';
import clsx from 'clsx';
import { api } from '@/lib/api';

interface StrategyBadgeProps {
    onClick?: () => void;
}

const STRATEGY_CONFIG: Record<string, { icon: React.ReactNode; label: string; color: string }> = {
    sliding_window: {
        icon: <Layers size={12} />,
        label: 'Sliding Window',
        color: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
    },
    token_budget: {
        icon: <Clock size={12} />,
        label: 'Token Budget',
        color: 'text-green-400 bg-green-500/10 border-green-500/20',
    },
    summarization: {
        icon: <FileText size={12} />,
        label: 'Summarization',
        color: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    },
};

export default function StrategyBadge({ onClick }: StrategyBadgeProps) {
    const [currentStrategy, setCurrentStrategy] = useState<string | null>(null);

    useEffect(() => {
        const fetchStrategy = async () => {
            try {
                const data = await api.getStrategies();
                setCurrentStrategy(data.current);
            } catch (err) {
                console.error('Failed to fetch strategy:', err);
            }
        };

        fetchStrategy();

        // Refresh every 30 seconds in case it changes
        const interval = setInterval(fetchStrategy, 30000);
        return () => clearInterval(interval);
    }, []);

    if (!currentStrategy) return null;

    const config = STRATEGY_CONFIG[currentStrategy] || STRATEGY_CONFIG.sliding_window;

    return (
        <button
            onClick={onClick}
            className={clsx(
                'flex items-center gap-1.5 px-2 py-1 text-xs font-medium rounded-full border transition-all',
                'hover:opacity-80 cursor-pointer',
                config.color
            )}
            title="Click to change strategy"
        >
            {config.icon}
            <span>{config.label}</span>
            <Settings size={10} className="opacity-50" />
        </button>
    );
}