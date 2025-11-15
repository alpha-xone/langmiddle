"""
SQLite storage backend implementation.

This module provides a local SQLite-based implementation of the chat storage interface.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import ChatStorageBackend, SortOrder, ThreadSortBy

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
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        custom_state TEXT DEFAULT '{}'::text
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
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            custom_state TEXT DEFAULT '{}'::text
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

    def authenticate(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """
        SQLite doesn't require authentication.

        Args:
            credentials: Ignored for SQLite

        Returns:
            Always True
        """
        return True

    def extract_user_id(self, credentials: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Extract user ID from credentials.

        Args:
            credentials: Dict containing 'user_id'

        Returns:
            User ID if provided
        """
        return credentials.get("user_id") if credentials else None

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

    def ensure_thread_exists(self, credentials: Dict[str, Any] | None, thread_id: str, user_id: str) -> bool:
        """
        Ensure chat thread exists in SQLite.

        Args:
            credentials: Authentication credentials (unused for SQLite)
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
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
        user_id: str,
        messages: List[AnyMessage],
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save messages to SQLite.

        Args:
            thread_id: Thread identifier
            user_id: User identifier
            messages: List of messages to save
            custom_state: Optional custom state defined in the graph

        Returns:
            Dict with 'saved_count' and 'errors' keys
        """
        saved_count = 0
        errors = []

        try:
            conn = self._persistent_conn if self._persistent_conn else None

            if self._persistent_conn:
                # Use persistent connection for in-memory database
                if not self.ensure_thread_exists(credentials, thread_id, user_id):
                    return {"saved_count": 0, "errors": ["Thread does not exist"]}

                if custom_state:
                    try:
                        self._persistent_conn.execute(
                            """
                            UPDATE chat_threads
                            SET metadata = ?
                            WHERE id = ?
                            """,
                            (json.dumps(custom_state), thread_id),
                        )
                        logger.debug(
                            f"Updated custom state for thread {thread_id} in SQLite database"
                        )
                    except Exception as e:
                        logger.error(f"Error updating custom state for thread {thread_id}: {e}")

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
                    if not self.ensure_thread_exists(credentials, thread_id, user_id):
                        return {"saved_count": 0, "errors": ["Thread does not exist"]}

                    if custom_state:
                        try:
                            conn.execute(
                                """
                                    UPDATE chat_threads
                                    SET metadata = ?
                                    WHERE id = ?
                                """,
                                (json.dumps(custom_state), thread_id),
                            )
                            logger.debug(
                                f"Updated custom state for thread {thread_id} in SQLite database"
                            )
                        except Exception as e:
                            logger.error(f"Error updating custom state for thread {thread_id}: {e}")

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

    def get_thread(
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
    ) -> dict | None:
        """
        Get a thread by ID.

        Args:
            thread_id: The ID of the thread to get.
        """
        # Fetch thread record
        try:
            if self._persistent_conn:
                cursor = self._persistent_conn.execute(
                    "SELECT id, user_id, created_at, updated_at, custom_state FROM chat_threads WHERE id = ?",
                    (thread_id,),
                )
                result = cursor.fetchone()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT id, user_id, created_at, updated_at, custom_state FROM chat_threads WHERE id = ?",
                        (thread_id,),
                    )
                    result = cursor.fetchone()
        except Exception as e:
            logger.error(f"Error executing thread query for id {thread_id}: {e}")
            return None

        if not result:
            return None

        # Fetch messages for this thread
        msgs = []
        try:
            if self._persistent_conn:
                cursor = self._persistent_conn.execute(
                    "SELECT id, content, role, created_at, metadata, usage_metadata FROM chat_messages WHERE thread_id = ? ORDER BY created_at ASC",
                    (thread_id,),
                )
                rows = cursor.fetchall()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT id, content, role, created_at, metadata, usage_metadata FROM chat_messages WHERE thread_id = ? ORDER BY created_at ASC",
                        (thread_id,),
                    )
                    rows = cursor.fetchall()

            for r in rows:
                msgs.append(
                    {
                        "id": r[0],
                        "content": r[1],
                        "role": r[2],
                        "created_at": r[3],
                        "metadata": json.loads(r[4]) if r[4] else None,
                        "usage_metadata": json.loads(r[5]) if r[5] else None,
                    }
                )
        except Exception as e:
            logger.error(f"Error fetching messages for thread {thread_id}: {e}")
            msgs = []

        return {
            "thread_id": result[0],
            "user_id": result[1],
            "created_at": result[2],
            "updated_at": result[3],
            "custom_state": json.loads(result[4]) if result[4] else None,
            "values": {"messages": msgs},
        }

    def search_threads(
        self,
        *,
        metadata: dict | None = None,
        values: dict | None = None,
        ids: List[str] | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: ThreadSortBy | None = "updated_at",
        sort_order: SortOrder | None = "desc",
    ) -> List[dict]:
        """
        Search for threads.

        Args:
            metadata: Thread metadata to filter on.
            values: State values to filter on.
            ids: List of thread IDs to filter by.
            limit: Limit on number of threads to return.
            offset: Offset in threads table to start search from.
            sort_by: Sort by field.
            sort_order: Sort order.

        Returns:
            list[dict]: List of the threads matching the search parameters.
        """
        try:
            # Build query dynamically
            query_parts = ["SELECT id, user_id, created_at, updated_at, custom_state FROM chat_threads"]
            params = []
            conditions = []

            # Filter by IDs if provided
            if ids:
                placeholders = ", ".join(["?"] * len(ids))
                conditions.append(f"id IN ({placeholders})")
                params.extend(ids)

            # Apply metadata filters (stored as JSON in custom_state)
            if metadata:
                for key, value in metadata.items():
                    # SQLite JSON support is limited, so we'll do a simple string search
                    # In production, you might want to use a more sophisticated approach
                    conditions.append("custom_state LIKE ?")
                    params.append(f'%"{key}":"{value}"%')

            # Add WHERE clause if we have conditions
            if conditions:
                query_parts.append("WHERE " + " AND ".join(conditions))

            # Add sorting
            if sort_by:
                direction = "DESC" if sort_order == "desc" else "ASC"
                query_parts.append(f"ORDER BY {sort_by} {direction}")

            # Add limit and offset
            query_parts.append("LIMIT ? OFFSET ?")
            params.extend([limit, offset])

            query = " ".join(query_parts)

            if self._persistent_conn:
                cursor = self._persistent_conn.execute(query, params)
                results = cursor.fetchall()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(query, params)
                    results = cursor.fetchall()

            if not results:
                return []

            thread_ids = [row[0] for row in results]
            msgs = []
            try:
                if thread_ids:
                    placeholders = ",".join(["?"] * len(thread_ids))
                    q = f"SELECT id, content, role, created_at, metadata, usage_metadata, thread_id FROM chat_messages WHERE thread_id IN ({placeholders}) ORDER BY created_at ASC"
                    if self._persistent_conn:
                        cursor = self._persistent_conn.execute(q, thread_ids)
                        msg_rows = cursor.fetchall()
                    else:
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.execute(q, thread_ids)
                            msg_rows = cursor.fetchall()

                    for r in msg_rows:
                        msgs.append({
                            "id": r[0],
                            "content": r[1],
                            "role": r[2],
                            "created_at": r[3],
                            "metadata": json.loads(r[4]) if r[4] else None,
                            "usage_metadata": json.loads(r[5]) if r[5] else None,
                            "thread_id": r[6],
                        })
            except Exception as e:
                logger.error(f"Error fetching messages for threads: {e}")
                msgs = []

            # Map messages to their threads
            msgs_by_thread: dict = {}
            for m in msgs:
                msgs_by_thread.setdefault(m.get("thread_id"), []).append(m)

            threads = []
            for row in results:
                thread_id = row[0]
                thread_info = {
                    "thread_id": thread_id,
                    "user_id": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "custom_state": json.loads(row[4]) if row[4] else None,
                    "values": {"messages": msgs_by_thread.get(thread_id, [])},
                }
                threads.append(thread_info)

            logger.debug(f"Found {len(threads)} threads matching search criteria")
            return threads

        except Exception as e:
            logger.error(f"Error searching threads: {e}")
            return []

    def delete_thread(
        self,
        thread_id: str,
    ):
        """
        Delete a thread.

        Args:
            thread_id: The ID of the thread to delete.

        Returns:
            None
        """
        # Delete messages first (due to foreign key constraint)
        if self._persistent_conn:
            try:
                self._persistent_conn.execute(
                    "DELETE FROM chat_messages WHERE thread_id = ?",
                    (thread_id,),
                )
            except Exception as e:
                logger.error(f"Error deleting messages for thread {thread_id}: {e}")
                return

            try:
                self._persistent_conn.execute(
                    "DELETE FROM chat_threads WHERE id = ?",
                    (thread_id,),
                )
                self._persistent_conn.commit()
                logger.info(f"Deleted thread {thread_id} and all its messages")
            except Exception as e:
                logger.error(f"Error deleting thread {thread_id}: {e}")
        else:
            # file-based DB
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "DELETE FROM chat_messages WHERE thread_id = ?",
                        (thread_id,),
                    )
            except Exception as e:
                logger.error(f"Error deleting messages for thread {thread_id}: {e}")
                return

            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "DELETE FROM chat_threads WHERE id = ?",
                        (thread_id,),
                    )
                    conn.commit()
                logger.info(f"Deleted thread {thread_id} and all its messages")
            except Exception as e:
                logger.error(f"Error deleting thread {thread_id}: {e}")

    # =========================================================================
    # Facts Management Methods - Not supported in SQLite backend
    # =========================================================================

    def get_or_create_embedding_table(self, dimension: int) -> bool:
        """Ensure an embedding table exists for the given dimension."""
        raise NotImplementedError(
            "Facts management not supported in SQLite backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def insert_facts(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: str,
        facts: Sequence[Dict[str, Any] | str],
        embeddings: Optional[List[List[float]]] = None,
        model_dimension: Optional[int] = None,
        cue_embeddings: Optional[List[List[tuple[str, List[float]]]]] = None,
    ) -> Dict[str, Any]:
        """Insert facts with optional embeddings and cue embeddings into storage."""
        raise NotImplementedError(
            "Facts management not supported in SQLite backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def query_facts(
        self,
        credentials: Optional[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None,
        model_dimension: Optional[int] = None,
        match_threshold: float = 0.75,
        match_count: int = 10,
        filter_namespaces: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Query facts using vector similarity search."""
        raise NotImplementedError(
            "Facts management not supported in SQLite backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def get_fact_by_id(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a fact by its ID."""
        raise NotImplementedError(
            "Facts management not supported in SQLite backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def update_fact(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
        updates: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update a fact's content and/or metadata."""
        raise NotImplementedError(
            "Facts management not supported in SQLite backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def delete_fact(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete a fact and its embeddings."""
        raise NotImplementedError(
            "Facts management not supported in SQLite backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def check_processed_message(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> bool:
        """Check if a message has already been processed."""
        raise NotImplementedError(
            "Processed messages tracking not supported in SQLite backend. "
            "Use SupabaseStorageBackend for facts support."
        )

    def mark_processed_message(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> bool:
        """Mark a message as processed."""
        raise NotImplementedError(
            "Processed messages tracking not supported in SQLite backend. "
            "Use SupabaseStorageBackend for facts support."
        )

    def check_processed_messages_batch(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Check which messages have already been processed (batch mode)."""
        raise NotImplementedError(
            "Processed messages tracking not supported in SQLite backend. "
            "Use SupabaseStorageBackend for facts support."
        )

    def mark_processed_messages_batch(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_data: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """Mark multiple messages as processed (batch mode)."""
        raise NotImplementedError(
            "Processed messages tracking not supported in SQLite backend. "
            "Use SupabaseStorageBackend for facts support."
        )
