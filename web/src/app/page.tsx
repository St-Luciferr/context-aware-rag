'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from '@/components/SideBar';
import ChatArea from '@/components/ChatArea';
import SettingsModal from '@/components/SettingsModal';
import { api, ApiError } from '@/lib/api';
import type { Session, Message, StatusResponse, Citation } from '@/types';

const MESSAGES_PER_PAGE = 20;

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoadingSessions, setIsLoadingSessions] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasMoreMessages, setHasMoreMessages] = useState(false);
  const [oldestTimestamp, setOldestTimestamp] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Refs to prevent concurrent calls
  const isLoadingMoreRef = useRef(false);
  const fetchInProgressRef = useRef(false);
  const lastFetchedSessionRef = useRef<string | null>(null);

  // Auto-dismiss errors after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Fetch status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await api.getStatus();
        setStatus(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch status:', err);
        setStatus({
          status: 'offline',
          model: 'Unknown',
          vector_store: '',
          collection: '',
        });
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch sessions
  const fetchSessions = useCallback(async () => {
    try {
      const data = await api.getSessions();
      setSessions(data.sessions);
    } catch (err) {
      console.error('Failed to fetch sessions:', err);
    } finally {
      setIsLoadingSessions(false);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Fetch messages when session changes
  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      setHasMoreMessages(false);
      setOldestTimestamp(null);
      lastFetchedSessionRef.current = null;
      return;
    }

    // Reset fetch-in-progress if session changed
    if (lastFetchedSessionRef.current !== activeSessionId) {
      fetchInProgressRef.current = false;
    }

    // Skip if already fetching for this session (prevents Strict Mode double-fetch)
    if (fetchInProgressRef.current && lastFetchedSessionRef.current === activeSessionId) {
      return;
    }

    let cancelled = false;
    fetchInProgressRef.current = true;
    lastFetchedSessionRef.current = activeSessionId;

    const fetchMessages = async () => {
      setIsLoadingMessages(true);
      try {
        const data = await api.getHistory(activeSessionId, MESSAGES_PER_PAGE);

        // Don't update state if this effect was cancelled
        if (cancelled) return;

        // Map messages with proper typing (include id for stable React keys)
        const formattedMessages: Message[] = data.messages.map((msg) => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
          citations: msg.citations,
        }));
        setMessages(formattedMessages);
        setHasMoreMessages(data.has_more);
        setOldestTimestamp(data.oldest_timestamp);
      } catch (err) {
        if (cancelled) return;
        console.error('Failed to fetch messages:', err);
        setMessages([]);
        setHasMoreMessages(false);
        setOldestTimestamp(null);
      } finally {
        if (!cancelled) {
          fetchInProgressRef.current = false;
          setIsLoadingMessages(false);
        }
      }
    };

    fetchMessages();

    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  // Load more messages (older ones)
  const handleLoadMore = useCallback(async () => {
    // Use both state and ref for guard - ref prevents rapid calls before state updates
    if (
      !activeSessionId ||
      !hasMoreMessages ||
      isLoadingMore ||
      isLoadingMoreRef.current ||
      !oldestTimestamp
    ) {
      return;
    }

    isLoadingMoreRef.current = true;
    setIsLoadingMore(true);
    try {
      const data = await api.getHistory(activeSessionId, MESSAGES_PER_PAGE, oldestTimestamp);
      const olderMessages: Message[] = data.messages.map((msg) => ({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp,
        citations: msg.citations,
      }));

      // Prepend older messages to the beginning
      setMessages((prev) => [...olderMessages, ...prev]);
      setHasMoreMessages(data.has_more);
      setOldestTimestamp(data.oldest_timestamp);
    } catch (err) {
      console.error('Failed to load more messages:', err);
    } finally {
      isLoadingMoreRef.current = false;
      setIsLoadingMore(false);
    }
  }, [activeSessionId, hasMoreMessages, isLoadingMore, oldestTimestamp]);

  // Create new session
  const handleNewSession = async () => {
    try {
      const data = await api.createSession();
      setActiveSessionId(data.session_id);
      setMessages([]);
      await fetchSessions();
    } catch (err) {
      console.error('Failed to create session:', err);
      setError('Failed to create new session');
    }
  };

  // Select session
  const handleSelectSession = (sessionId: string) => {
    setActiveSessionId(sessionId);
  };

  // Delete session
  const handleDeleteSession = async (sessionId: string) => {
    try {
      await api.deleteSession(sessionId);
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
      await fetchSessions();
    } catch (err) {
      console.error('Failed to delete session:', err);
      setError('Failed to delete session');
    }
  };

  // Send message
  const handleSendMessage = async (content: string) => {
    let currentSessionId = activeSessionId;

    // Create new session if none active
    if (!currentSessionId) {
      try {
        const data = await api.createSession();
        currentSessionId = data.session_id;
        setActiveSessionId(currentSessionId);
      } catch (err) {
        console.error('Failed to create session:', err);
        setError('Failed to create session');
        return;
      }
    }

    // Optimistically add user message with timestamp
    const userMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsSending(true);

    try {
      const data = await api.sendMessage({
        message: content,
        session_id: currentSessionId,
      });

      // Add assistant message with citations
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
        citations: data.citations,
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Refresh sessions to update preview
      await fetchSessions();
    } catch (err) {
      console.error('Failed to send message:', err);
      // Remove optimistic message on error
      setMessages((prev) => prev.slice(0, -1));
      setError(
        err instanceof ApiError
          ? err.message
          : 'Failed to send message. Please try again.'
      );
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="flex h-screen bg-dark-950 text-white">
      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />

      {/* Error toast */}
      {error && (
        <div className="fixed top-4 right-4 z-50 max-w-md animate-fade-in">
          <div className="bg-red-500/90 backdrop-blur-sm text-white px-4 py-3 rounded-lg shadow-lg border border-red-400/20">
            <div className="flex items-center gap-3">
              <span className="flex-1">{error}</span>
              <button
                onClick={() => setError(null)}
                className="text-white/80 hover:text-white text-lg leading-none"
              >
                ×
              </button>
            </div>
          </div>
        </div>
      )}

      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        status={status}
        isLoading={isLoadingSessions}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      <main className="flex-1 flex flex-col min-w-0">
        <ChatArea
          messages={messages}
          isLoading={isSending || isLoadingMessages}
          isLoadingMore={isLoadingMore}
          hasMoreMessages={hasMoreMessages}
          onSendMessage={handleSendMessage}
          onLoadMore={handleLoadMore}
          onOpenSettings={() => setIsSettingsOpen(true)}
        />
      </main>
    </div>
  );
}