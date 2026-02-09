#  chat_storage.py

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import json
import os


class ChatStorage:
    """Manages SQLite database for session persistence."""

    def __init__(self, db_path: str = "data/chat_sessions.db"):
        cwd = Path().resolve()
        print(f"Current working directory: {cwd}")
        self.db_path = cwd.joinpath(db_path)
        print(f"DB Path: {self.db_path}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path_obj = Path(self.db_path)
        if db_path_obj.exists() and db_path_obj.is_dir():
            os.remove(db_path_obj)
            print(
                f"Database path '{db_path}' exists and is a directory. "
                "Remove or rename that directory and create a file instead."
            )

        try:
            if not db_path_obj.exists():
                db_path_obj.touch(exist_ok=False)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to create DB file '{db_path_obj}': {exc}") from exc

        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_message_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id, timestamp)
            """)

            conn.commit()

    def create_session(self, session_id: str):
        """Create a new session."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO sessions (session_id, created_at, last_message_at, metadata)
                VALUES (?, ?, ?, ?)
            """, (session_id, now, now, json.dumps({})))
            conn.commit()

    def update_session(self, session_id: str):
        """Update session's last message timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions SET last_message_at = ? WHERE session_id = ?
            """, (now, session_id))
            conn.commit()

    def save_message(self, session_id: str, role: str, content: str, citations: list[dict] = None):
        """Save a message to the database."""
        now = datetime.now(timezone.utc).isoformat()
        citations_json = json.dumps(citations) if citations else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, citations, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, content, citations_json, now))
            conn.commit()

    def load_messages(self, session_id: str) -> list[dict]:
        """Load all messages for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, citations, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))

            messages = []
            for row in cursor.fetchall():
                msg = {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[3]
                }
                if row[2]:  # citations
                    msg["citations"] = json.loads(row[2])
                messages.append(msg)

            return messages

    def load_messages_paginated(
        self,
        session_id: str,
        limit: int = 20,
        before_timestamp: str = None
    ) -> dict:
        """Load messages for a session with pagination (newest first, for infinite scroll up).

        Args:
            session_id: The session ID
            limit: Maximum number of messages to return
            before_timestamp: Only return messages before this timestamp (for pagination)

        Returns:
            Dict with messages, has_more flag, and oldest_timestamp for next page
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get total count for this session
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,)
            )
            total_count = cursor.fetchone()[0]

            # Build query with optional before_timestamp filter (include id for stable React keys)
            if before_timestamp:
                cursor.execute("""
                    SELECT id, role, content, citations, timestamp
                    FROM messages
                    WHERE session_id = ? AND timestamp < ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (session_id, before_timestamp, limit))
            else:
                cursor.execute("""
                    SELECT id, role, content, citations, timestamp
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (session_id, limit))

            messages = []
            for row in cursor.fetchall():
                msg = {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[4]
                }
                if row[3]:  # citations
                    msg["citations"] = json.loads(row[3])
                messages.append(msg)

            # Reverse to get chronological order (oldest first)
            messages.reverse()

            # Determine if there are more messages
            oldest_timestamp = messages[0]["timestamp"] if messages else None

            if oldest_timestamp:
                cursor.execute("""
                    SELECT COUNT(*) FROM messages
                    WHERE session_id = ? AND timestamp < ?
                """, (session_id, oldest_timestamp))
                remaining = cursor.fetchone()[0]
                has_more = remaining > 0
            else:
                has_more = False

            return {
                "messages": messages,
                "has_more": has_more,
                "total_count": total_count,
                "oldest_timestamp": oldest_timestamp
            }

    def get_session_info(self, session_id: str) -> dict:
        """Get session metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT created_at, last_message_at, 
                       (SELECT COUNT(*) FROM messages WHERE session_id = ?) as message_count
                FROM sessions 
                WHERE session_id = ?
            """, (session_id, session_id))

            row = cursor.fetchone()
            if row:
                return {
                    "session_id": session_id,
                    "created_at": row[0],
                    "last_message_at": row[1],
                    "message_count": row[2]
                }
            return None

    def get_all_sessions(self) -> list[dict]:
        """Get all session information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.session_id, s.created_at, s.last_message_at,
                       COUNT(m.id) as message_count,
                       (SELECT content FROM messages 
                        WHERE session_id = s.session_id 
                        ORDER BY timestamp DESC LIMIT 1) as last_message
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.last_message_at DESC
            """)

            sessions = []
            for row in cursor.fetchall():
                preview = None
                if row[4]:  # last_message
                    preview = row[4][:100] + \
                        "..." if len(row[4]) > 100 else row[4]

                sessions.append({
                    "session_id": row[0],
                    "created_at": row[1],
                    "last_message_at": row[2],
                    "message_count": row[3],
                    "preview": preview
                })

            return sessions

    def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            return cursor.fetchone() is not None
