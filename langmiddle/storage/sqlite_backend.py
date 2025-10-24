"""
SQLite storage backend implementation.

This module provides a local SQLite-based implementation of the chat storage interface.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import ChatStorageBackend

logger = get_graph_logger(__name__)

__all__ = ["SQLiteStorageBackend"]


class SQLiteStorageBackend(ChatStorageBackend):
    """SQLite implementation of chat storage backend."""

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize SQLite storage backend.

        Args:
            db_path: Path to SQLite database file (use ":memory:" for in-memory database)
        """
        self.db_path = db_path if db_path == ":memory:" else str(Path(db_path))

        # For in-memory databases, maintain a persistent connection
        self._persistent_conn = None
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )

        self._init_database()

    def _get_connection(self):
        """Get database connection (persistent for in-memory, new for file-based)."""
        if self._persistent_conn:
            return self._persistent_conn
        return sqlite3.connect(self.db_path)

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            conn = self._get_connection()
            with_context = (
                conn if self._persistent_conn else sqlite3.connect(self.db_path)
            )

            if self._persistent_conn:
                # Use persistent connection directly
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_threads (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        role TEXT NOT NULL,
                        metadata TEXT,
                        usage_metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (thread_id) REFERENCES chat_threads (id)
                    )
                """
                )
                conn.commit()
            else:
                # Use context manager for file-based database
                with with_context as conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_threads (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """
                    )

                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_messages (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            thread_id TEXT NOT NULL,
                            content TEXT NOT NULL,
                            role TEXT NOT NULL,
                            metadata TEXT,
                            usage_metadata TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (thread_id) REFERENCES chat_threads (id)
                        )
                    """
                    )
                    conn.commit()

            logger.debug(f"SQLite database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        SQLite doesn't require authentication.

        Args:
            credentials: Ignored for SQLite

        Returns:
            Always True
        """
        return True

    def extract_user_id(self, credentials: Dict[str, Any]) -> Optional[str]:
        """
        Extract user ID from credentials.

        Args:
            credentials: Dict containing 'user_id'

        Returns:
            User ID if provided
        """
        return credentials.get("user_id")

    def get_existing_message_ids(self, thread_id: str) -> set:
        """
        Get existing message IDs from SQLite.

        Args:
            thread_id: Thread identifier

        Returns:
            Set of existing message IDs
        """
        try:
            if self._persistent_conn:
                cursor = self._persistent_conn.execute(
                    "SELECT id FROM chat_messages WHERE thread_id = ?", (thread_id,)
                )
                message_ids = {row[0] for row in cursor.fetchall()}
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT id FROM chat_messages WHERE thread_id = ?", (thread_id,)
                    )
                    message_ids = {row[0] for row in cursor.fetchall()}

            logger.debug(
                f"Found {len(message_ids)} existing messages for thread {thread_id}"
            )
            return message_ids
        except Exception as e:
            logger.error(f"Error fetching existing messages: {e}")
            return set()

    def ensure_thread_exists(self, thread_id: str, user_id: str) -> bool:
        """
        Ensure chat thread exists in SQLite.

        Args:
            thread_id: Thread identifier
            user_id: User identifier

        Returns:
            True if thread exists or was created
        """
        try:
            if self._persistent_conn:
                self._persistent_conn.execute(
                    "INSERT OR REPLACE INTO chat_threads (id, user_id) VALUES (?, ?)",
                    (thread_id, user_id),
                )
                self._persistent_conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO chat_threads (id, user_id) VALUES (?, ?)",
                        (thread_id, user_id),
                    )
                    conn.commit()

            logger.debug(f"Chat thread {thread_id} ensured in SQLite database")
            return True
        except Exception as e:
            logger.error(f"Error ensuring thread exists: {e}")
            return False

    def save_messages(
        self, thread_id: str, user_id: str, messages: List[AnyMessage]
    ) -> Dict[str, Any]:
        """
        Save messages to SQLite.

        Args:
            thread_id: Thread identifier
            user_id: User identifier
            messages: List of messages to save

        Returns:
            Dict with 'saved_count' and 'errors' keys
        """
        saved_count = 0
        errors = []

        try:
            conn = self._persistent_conn if self._persistent_conn else None

            if self._persistent_conn:
                # Use persistent connection for in-memory database
                for msg in messages:
                    try:
                        self._persistent_conn.execute(
                            """
                            INSERT OR REPLACE INTO chat_messages
                            (id, user_id, thread_id, content, role, metadata, usage_metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                msg.id,
                                user_id,
                                thread_id,
                                msg.content,
                                self.TYPE_TO_ROLE.get(msg.type, msg.type),
                                json.dumps(getattr(msg, "response_metadata", {})),
                                json.dumps(getattr(msg, "usage_metadata", {})),
                            ),
                        )
                        saved_count += 1
                        logger.debug(f"Saved message {msg.id} to SQLite database")
                    except Exception as e:
                        errors.append(f"Error saving message {msg.id}: {e}")
                        logger.error(f"Error saving message {msg.id}: {e}")

                self._persistent_conn.commit()
            else:
                # Use context manager for file-based database
                with sqlite3.connect(self.db_path) as conn:
                    for msg in messages:
                        try:
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO chat_messages
                                (id, user_id, thread_id, content, role, metadata, usage_metadata)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    msg.id,
                                    user_id,
                                    thread_id,
                                    msg.content,
                                    self.TYPE_TO_ROLE.get(msg.type, msg.type),
                                    json.dumps(getattr(msg, "response_metadata", {})),
                                    json.dumps(getattr(msg, "usage_metadata", {})),
                                ),
                            )
                            saved_count += 1
                            logger.debug(f"Saved message {msg.id} to SQLite database")
                        except Exception as e:
                            errors.append(f"Error saving message {msg.id}: {e}")
                            logger.error(f"Error saving message {msg.id}: {e}")

                    conn.commit()

        except Exception as e:
            errors.append(f"SQLite database error: {e}")
            logger.error(f"SQLite database error: {e}")

        return {"saved_count": saved_count, "errors": errors}
