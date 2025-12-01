'use client';

import { useState, useEffect, useCallback } from 'react';
import Sidebar from '@/components/SideBar';
import ChatArea from '@/components/ChatArea';
import { api, ApiError } from '@/lib/api';
import type { Session, Message, StatusResponse } from '@/types';

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoadingSessions, setIsLoadingSessions] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await api.getStatus();
        setStatus(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch status:', err);
        setStatus({ status: 'offline', model: 'Unknown', vector_store: '', collection: '' });
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
    const fetchMessages = async () => {
      if (!activeSessionId) {
        setMessages([]);
        return;
      }

      setIsLoadingMessages(true);
      try {
        const data = await api.getHistory(activeSessionId);
        setMessages(data.messages);
      } catch (err) {
        console.error('Failed to fetch messages:', err);
        setMessages([]);
      } finally {
        setIsLoadingMessages(false);
      }
    };

    fetchMessages();
  }, [activeSessionId]);

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

    // Optimistically add user message
    const userMessage: Message = { role: 'user', content };
    setMessages((prev) => [...prev, userMessage]);
    setIsSending(true);

    try {
      const data = await api.sendMessage({
        message: content,
        session_id: currentSessionId,
      });

      // Add assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
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
      {/* Error toast */}
      {error && (
        <div className="fixed top-4 right-4 z-50 bg-red-500/90 text-white px-4 py-3 rounded-lg shadow-lg animate-fade-in">
          <div className="flex items-center gap-3">
            <span>{error}</span>
            <button
              onClick={() => setError(null)}
              className="text-white/80 hover:text-white"
            >
              ✕
            </button>
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
      />

      <main className="flex-1 flex flex-col min-w-0">
        <ChatArea
          messages={messages}
          isLoading={isSending || isLoadingMessages}
          onSendMessage={handleSendMessage}
        />
      </main>
    </div>
  );
}