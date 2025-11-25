"""
SQLite storage backend implementation.

This module provides a local SQLite-based implementation of the chat storage interface.
"""

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import ChatStorageBackend, SortOrder, ThreadSortBy

logger = get_graph_logger(__name__)

# Try to import sqlite-vec
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    sqlite_vec = None
    SQLITE_VEC_AVAILABLE = False
    logger.warning(
        "sqlite-vec not available. Install with: pip install sqlite-vec"
    )

__all__ = ["SQLiteStorageBackend", "SQLITE_VEC_AVAILABLE"]


class SQLiteStorageBackend(ChatStorageBackend):
    """SQLite implementation of chat storage backend."""

    def __init__(
        self,
        db_path: str = ":memory:",
        auto_create_tables: bool = False,
        enable_facts: bool = False
    ):
        """
        Initialize SQLite storage backend.

        Args:
            db_path: Path to SQLite database file (use ":memory:" for in-memory database)
            auto_create_tables: Whether to automatically create tables
            enable_facts: Whether to enable facts management (requires sqlite-vec)
        """
        self.db_path = db_path if db_path == ":memory:" else str(Path(db_path))
        self.enable_facts = enable_facts

        # For in-memory databases, maintain a persistent connection
        self._persistent_conn = None
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._persistent_conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension if facts are enabled
        if self.enable_facts:
            if not SQLITE_VEC_AVAILABLE:
                raise RuntimeError(
                    "Facts management requires sqlite-vec. "
                    "Install with: pip install 'langmiddle[sqlite]'"
                )
            conn = self._get_connection()
            conn.enable_load_extension(True)
            if SQLITE_VEC_AVAILABLE and sqlite_vec is not None:
                sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            if not self._persistent_conn:
                conn.close()

        self._init_database()

        if auto_create_tables and enable_facts:
            self._init_facts_tables()

    def _get_connection(self):
        """Get database connection (persistent for in-memory, new for file-based)."""
        if self._persistent_conn:
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension for each new connection if facts enabled
        if self.enable_facts and SQLITE_VEC_AVAILABLE:
            conn.enable_load_extension(True)
            if SQLITE_VEC_AVAILABLE and sqlite_vec is not None:
                sqlite_vec.load(conn)
            conn.enable_load_extension(False)

        return conn

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
                        custom_state TEXT DEFAULT '{}'
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
                            custom_state TEXT DEFAULT '{}'
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

    def _init_facts_tables(self):
        """Initialize facts-related tables from SQL file."""
        try:
            # Load SQL from file
            sql_file = Path(__file__).parent / "sqlite" / "chat_facts.sql"
            if not sql_file.exists():
                logger.warning(f"Facts SQL file not found at {sql_file}")
                return

            with open(sql_file, 'r') as f:
                sql_script = f.read()

            conn = self._get_connection()
            if self._persistent_conn:
                conn.executescript(sql_script)
                conn.commit()
            else:
                with conn:
                    conn.executescript(sql_script)
                conn.close()

            logger.debug("Facts tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize facts tables: {e}")
            raise

    def authenticate(self, credentials: Dict[str, Any] | None) -> bool:
        """
        SQLite doesn't require authentication.

        Args:
            credentials: Optional authentication credentials (ignored for SQLite)

        Returns:
            Always True
        """
        return True

    def extract_user_id(self, credentials: Dict[str, Any] | None) -> str | None:
        """
        Extract user ID from credentials.

        Args:
            credentials: Authentication credentials containing 'user_id'

        Returns:
            User ID if provided
        """
        if not credentials or not isinstance(credentials, dict):
            return None
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

    def ensure_thread_exists(self, thread_id: str, user_id: str, credentials: Dict[str, Any] | None = None) -> bool:
        """
        Ensure chat thread exists in SQLite.

        Args:
            thread_id: Thread identifier
            user_id: User identifier
            credentials: Optional authentication credentials (ignored for SQLite)

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
        thread_id: str,
        user_id: str,
        messages: List[AnyMessage],
        custom_state: Dict[str, Any] | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Save messages to SQLite.

        Args:
            thread_id: Thread identifier
            user_id: User identifier
            messages: List of messages to save
            custom_state: Optional custom state defined in the graph
            credentials: Optional authentication credentials (ignored for SQLite)

        Returns:
            Dict with 'saved_count' and 'errors' keys
        """
        saved_count = 0
        errors = []

        try:
            conn = self._persistent_conn if self._persistent_conn else None

            if self._persistent_conn:
                # Use persistent connection for in-memory database
                if not self.ensure_thread_exists(thread_id=thread_id, user_id=user_id):
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
                        # Prepare metadata with additional_kwargs if present
                        metadata = getattr(msg, "response_metadata", {})
                        if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                            metadata = metadata.copy() if metadata else {}
                            metadata.update(msg.additional_kwargs)

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
                                json.dumps(metadata),
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
                    if not self.ensure_thread_exists(thread_id=thread_id, user_id=user_id):
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
                            # Prepare metadata with additional_kwargs if present
                            metadata = getattr(msg, "response_metadata", {})
                            if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                                metadata = metadata.copy() if metadata else {}
                                metadata.update(msg.additional_kwargs)

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
                                    json.dumps(metadata),
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
        thread_id: str,
        credentials: Dict[str, Any] | None = None,
    ) -> dict | None:
        """
        Get a thread by ID.

        Args:
            thread_id: The ID of the thread to get.
            credentials: Optional authentication credentials (ignored for SQLite)
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
    # Facts Management Methods - SQLite with sqlite-vec
    # =========================================================================

    def get_or_create_embedding_table(self, dimension: int) -> bool:
        """Ensure an embedding table exists for the given dimension."""
        if not self.enable_facts:
            raise RuntimeError(
                "Facts management not enabled. "
                "Initialize with enable_facts=True"
            )

        table_name = f"fact_embeddings_{dimension}"
        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?
                """,
                (table_name,)
            )

            if cursor.fetchone():
                if not self._persistent_conn:
                    conn.close()
                return True

            # Create sqlite-vec virtual table
            # Note: Virtual tables cannot be indexed, but metadata columns
            # with + prefix can be used in WHERE clauses
            cursor.execute(f"""
                CREATE VIRTUAL TABLE {table_name} USING vec0(
                    embedding FLOAT[{dimension}],
                    +fact_id TEXT NOT NULL,
                    +user_id TEXT NOT NULL,
                    +created_at TEXT NOT NULL
                )
            """)

            if self._persistent_conn:
                conn.commit()
            else:
                conn.commit()
                conn.close()

            logger.debug(f"Created embedding table: {table_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create embedding table {table_name}: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                conn.close()
            return False

    def insert_facts(
        self,
        user_id: str,
        facts: Sequence[Dict[str, Any] | str],
        embeddings: List[List[float]] | None = None,
        model_dimension: int | None = None,
        cue_embeddings: List[List[tuple[str, List[float]]]] | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Insert facts with optional embeddings into storage."""
        if not self.enable_facts:
            return {
                "success": False,
                "inserted_count": 0,
                "fact_ids": [],
                "errors": ["Facts management not enabled"]
            }

        # Infer model_dimension from embeddings if not provided
        if embeddings and not model_dimension:
            if embeddings and embeddings[0]:
                model_dimension = len(embeddings[0])
                logger.debug(f"Inferred model_dimension={model_dimension} from embeddings")

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            fact_ids = []
            errors = []

            # Ensure embedding table exists if embeddings provided
            if embeddings and model_dimension:
                self.get_or_create_embedding_table(model_dimension)

            for i, fact in enumerate(facts):
                try:
                    # Parse fact content
                    if isinstance(fact, str):
                        content = fact
                        namespace = []
                        language = "en"
                        intensity = None
                        confidence = None
                    else:
                        content = fact["content"]
                        namespace = fact.get("namespace", [])
                        language = fact.get("language", "en")
                        intensity = fact.get("intensity")
                        confidence = fact.get("confidence")

                    # Insert into facts table
                    cursor.execute("""
                        INSERT INTO facts (
                            user_id, content, namespace, language,
                            intensity, confidence, model_dimension
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id,
                        content,
                        json.dumps(namespace),
                        language,
                        intensity,
                        confidence,
                        model_dimension or 0
                    ))

                    fact_id = cursor.lastrowid
                    # SQLite returns integer rowid, convert to hex string for consistency
                    cursor.execute("SELECT id FROM facts WHERE rowid = ?", (fact_id,))
                    fact_id_str = cursor.fetchone()[0]
                    fact_ids.append(fact_id_str)

                    # Insert embedding if provided
                    if embeddings and i < len(embeddings) and model_dimension:
                        embedding = embeddings[i]
                        table_name = f"fact_embeddings_{model_dimension}"

                        # Serialize embedding as JSON array for sqlite-vec
                        embedding_json = json.dumps(embedding)

                        cursor.execute(f"""
                            INSERT INTO {table_name} (
                                embedding, fact_id, user_id, created_at
                            ) VALUES (?, ?, ?, datetime('now'))
                        """, (embedding_json, fact_id_str, user_id))

                except Exception as e:
                    errors.append(f"Error inserting fact {i}: {str(e)}")
                    logger.error(f"Error inserting fact {i}: {e}")

            if self._persistent_conn:
                conn.commit()
            else:
                conn.commit()
                conn.close()

            return {
                "success": len(fact_ids) > 0,
                "inserted_count": len(fact_ids),
                "fact_ids": fact_ids,
                "errors": errors
            }

        except Exception as e:
            logger.error(f"Failed to insert facts: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                conn.close()
            return {
                "success": False,
                "inserted_count": 0,
                "fact_ids": [],
                "errors": [str(e)]
            }

    def query_facts(
        self,
        query_embedding: List[float] | None = None,
        user_id: str | None = None,
        model_dimension: int | None = None,
        match_threshold: float = 0.75,
        match_count: int = 10,
        filter_namespaces: List[List[str]] | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Query facts using vector similarity search with sqlite-vec.

        If query_embedding is None, lists all facts (optionally filtered by namespace).
        """
        if not self.enable_facts:
            return []

        # Infer model_dimension from query_embedding if not provided
        if query_embedding and not model_dimension:
            model_dimension = len(query_embedding)
            logger.debug(f"Inferred model_dimension={model_dimension} from query_embedding")

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if match_count is None:
                match_count = 10

            # If no query_embedding, list all facts (no similarity search)
            if query_embedding is None:
                # Build query with namespace filter if provided
                query = """
                    SELECT
                        id, content, namespace, language, intensity, confidence,
                        model_dimension, created_at, updated_at, access_count,
                        last_accessed_at, relevance_score
                    FROM facts
                    WHERE user_id = ?
                """
                params = [user_id]

                # Add namespace filter if specified
                if filter_namespaces:
                    # SQLite doesn't have array operations, so we filter in Python
                    cursor.execute(query + " LIMIT ?", params + [match_count * 10])
                else:
                    cursor.execute(query + " LIMIT ?", params + [match_count])

                rows = cursor.fetchall()

                if not self._persistent_conn:
                    conn.close()

                if not rows:
                    logger.debug(f"No facts found for user_id={user_id}")
                    return []

                results = []
                for row in rows:
                    try:
                        namespace = json.loads(row[2])
                    except Exception:
                        namespace = []

                    # Filter by namespaces if specified
                    if filter_namespaces:
                        namespace_match = False
                        for ns_filter in filter_namespaces:
                            if len(namespace) >= len(ns_filter):
                                if namespace[:len(ns_filter)] == ns_filter:
                                    namespace_match = True
                                    break
                        if not namespace_match:
                            continue

                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "namespace": namespace,
                        "language": row[3],
                        "intensity": row[4],
                        "confidence": row[5],
                        "model_dimension": row[6],
                        "created_at": row[7],
                        "updated_at": row[8],
                        "access_count": row[9],
                        "last_accessed_at": row[10],
                        "relevance_score": row[11] or 0.5,
                    })

                    if len(results) >= match_count:
                        break

                logger.info(f"Listed {len(results)} facts for user_id={user_id}")
                return results

            # Vector similarity search mode
            if not model_dimension:
                logger.error("model_dimension required when query_embedding is provided")
                if not self._persistent_conn:
                    conn.close()
                return []

            table_name = f"fact_embeddings_{model_dimension}"

            # Check if embedding table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?
            """, (table_name,))

            if not cursor.fetchone():
                logger.warning(f"Embedding table {table_name} does not exist")
                if not self._persistent_conn:
                    conn.close()
                return []

            # Serialize query embedding as JSON
            query_json = json.dumps(query_embedding)

            # Use sqlite-vec's vector search
            # Note: sqlite-vec is very particular about query format
            # Must use: WHERE embedding MATCH ? AND k = ? ONLY
            # Cannot join or use auxiliary columns in WHERE
            cursor.execute(f"""
                SELECT
                    rowid,
                    fact_id,
                    user_id,
                    distance
                FROM {table_name}
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
            """, (query_json, match_count * 2))  # Get more results to filter

            # Get fact IDs and distances
            embedding_results = cursor.fetchall()

            if not embedding_results:
                if not self._persistent_conn:
                    conn.close()
                return []

            # Now fetch full fact details from facts table
            fact_ids = [row[1] for row in embedding_results]
            placeholders = ','.join('?' * len(fact_ids))

            cursor.execute(f"""
                SELECT
                    id, content, namespace, language, intensity, confidence,
                    model_dimension, created_at, updated_at, access_count,
                    last_accessed_at, relevance_score
                FROM facts
                WHERE id IN ({placeholders})
            """, fact_ids)            # Build facts lookup
            facts_map = {}
            for row in cursor.fetchall():
                facts_map[row[0]] = row

            results = []
            seen_facts = set()

            # Process embedding results with fact details
            for emb_row in embedding_results:
                fact_id = emb_row[1]
                row_user_id = emb_row[2]
                distance = emb_row[3]

                # Filter by user_id (must be done in Python for sqlite-vec)
                if user_id and row_user_id != user_id:
                    continue

                # Deduplicate (same fact might have multiple embeddings)
                if fact_id in seen_facts:
                    continue

                # Get fact details
                if fact_id not in facts_map:
                    continue

                fact = facts_map[fact_id]

                # Calculate similarity from distance (L2 distance)
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

                # Apply threshold
                if similarity < match_threshold:
                    continue

                # Parse namespace JSON
                try:
                    namespace = json.loads(fact[2])
                except Exception:
                    namespace = []

                # Filter by namespaces if specified
                if filter_namespaces:
                    namespace_match = False
                    for ns_filter in filter_namespaces:
                        if len(namespace) >= len(ns_filter):
                            if namespace[:len(ns_filter)] == ns_filter:
                                namespace_match = True
                                break
                    if not namespace_match:
                        continue

                # Calculate combined score (70% similarity + 30% relevance)
                relevance_score = fact[11] or 0.5
                combined_score = (similarity * 0.7) + (relevance_score * 0.3)

                results.append({
                    "id": fact_id,
                    "content": fact[1],
                    "namespace": namespace,
                    "language": fact[3],
                    "intensity": fact[4],
                    "confidence": fact[5],
                    "model_dimension": fact[6],
                    "created_at": fact[7],
                    "updated_at": fact[8],
                    "access_count": fact[9],
                    "last_accessed_at": fact[10],
                    "relevance_score": relevance_score,
                    "similarity": similarity,
                    "combined_score": combined_score
                })

                seen_facts.add(fact_id)

                if len(results) >= match_count:
                    break

            # Sort by combined score
            results.sort(key=lambda x: x["combined_score"], reverse=True)

            if not self._persistent_conn:
                conn.close()

            return results[:match_count]

        except Exception as e:
            logger.error(f"Failed to query facts: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return []

    def get_fact_by_id(
        self,
        fact_id: str,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """Get a fact by its ID."""
        if not self.enable_facts:
            return None

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    id, content, namespace, language, intensity, confidence,
                    model_dimension, created_at, updated_at,
                    access_count, last_accessed_at, relevance_score
                FROM facts
                WHERE id = ? AND user_id = ?
            """, (fact_id, user_id))

            row = cursor.fetchone()

            if not self._persistent_conn:
                conn.close()

            if not row:
                return None

            try:
                namespace = json.loads(row[2])
            except Exception:
                namespace = []

            return {
                "id": row[0],
                "content": row[1],
                "namespace": namespace,
                "language": row[3],
                "intensity": row[4],
                "confidence": row[5],
                "model_dimension": row[6],
                "created_at": row[7],
                "updated_at": row[8],
                "access_count": row[9],
                "last_accessed_at": row[10],
                "relevance_score": row[11]
            }

        except Exception as e:
            logger.error(f"Failed to get fact {fact_id}: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return None

    def update_fact(
        self,
        fact_id: str,
        user_id: str | None = None,
        updates: Dict[str, Any] | None = None,
        embedding: List[float] | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """Update a fact's content and/or metadata."""
        if not self.enable_facts or not updates:
            return False

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build update query
            set_clauses = []
            params = []

            if "content" in updates:
                set_clauses.append("content = ?")
                params.append(updates["content"])

            if "namespace" in updates:
                set_clauses.append("namespace = ?")
                params.append(json.dumps(updates["namespace"]))

            if "language" in updates:
                set_clauses.append("language = ?")
                params.append(updates["language"])

            if "intensity" in updates:
                set_clauses.append("intensity = ?")
                params.append(updates["intensity"])

            if "confidence" in updates:
                set_clauses.append("confidence = ?")
                params.append(updates["confidence"])

            # Always update updated_at timestamp
            set_clauses.append("updated_at = ?")
            params.append(datetime.now().isoformat())

            if not set_clauses:
                return False

            params.extend([fact_id, user_id])

            cursor.execute(f"""
                UPDATE facts
                SET {', '.join(set_clauses)}
                WHERE id = ? AND user_id = ?
            """, params)

            # Update embedding if provided
            if embedding:
                # Get model dimension
                cursor.execute(
                    "SELECT model_dimension FROM facts WHERE id = ?",
                    (fact_id,)
                )
                row = cursor.fetchone()
                if row:
                    model_dimension = row[0]
                    table_name = f"fact_embeddings_{model_dimension}"

                    # Delete old embedding
                    try:
                        cursor.execute(f"""
                            DELETE FROM {table_name}
                            WHERE fact_id = ? AND user_id = ?
                        """, (fact_id, user_id))
                    except Exception as del_err:
                        logger.error(f"Delete error: {del_err}")
                        raise

                    # Insert new embedding
                    embedding_json = json.dumps(embedding)
                    # Note: Use all columns in order for vec0 virtual table
                    try:
                        cursor.execute(f"""
                            INSERT INTO {table_name} (embedding, fact_id, user_id, created_at)
                            VALUES (?, ?, ?, ?)
                        """, (embedding_json, fact_id, user_id, datetime.now().isoformat()))
                    except Exception as ins_err:
                        logger.error(f"Insert error: {ins_err}")
                        raise

            if self._persistent_conn:
                conn.commit()
            else:
                conn.commit()
                conn.close()

            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to update fact {fact_id}: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return False

    def delete_fact(
        self,
        fact_id: str,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """Delete a fact and its embeddings."""
        if not self.enable_facts:
            return False

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get model dimension to delete embeddings
            cursor.execute(
                "SELECT model_dimension FROM facts WHERE id = ? AND user_id = ?",
                (fact_id, user_id)
            )
            row = cursor.fetchone()

            if row:
                model_dimension = row[0]
                table_name = f"fact_embeddings_{model_dimension}"

                # Delete embeddings
                cursor.execute(f"""
                    DELETE FROM {table_name}
                    WHERE fact_id = ? AND user_id = ?
                """, (fact_id, user_id))

            # Delete fact (cascade will handle access logs via trigger)
            cursor.execute("""
                DELETE FROM facts
                WHERE id = ? AND user_id = ?
            """, (fact_id, user_id))

            if self._persistent_conn:
                conn.commit()
            else:
                conn.commit()
                conn.close()

            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to delete fact {fact_id}: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return False

    def check_processed_message(
        self,
        user_id: str | None = None,
        message_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """Check if a message has already been processed."""
        if not self.enable_facts or not user_id or not message_id:
            return False

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 1 FROM processed_messages
                WHERE user_id = ? AND message_id = ?
                LIMIT 1
            """, (user_id, message_id))

            result = cursor.fetchone() is not None

            if not self._persistent_conn:
                conn.close()

            return result

        except Exception as e:
            logger.error(f"Failed to check processed message: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return False

    def mark_processed_message(
        self,
        user_id: str | None = None,
        message_id: str | None = None,
        thread_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """Mark a message as processed."""
        if not self.enable_facts or not user_id or not message_id:
            return False

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR IGNORE INTO processed_messages (user_id, message_id, thread_id)
                VALUES (?, ?, ?)
            """, (user_id, message_id, thread_id))

            if self._persistent_conn:
                conn.commit()
            else:
                conn.commit()
                conn.close()

            return True

        except Exception as e:
            logger.error(f"Failed to mark processed message: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return False

    def check_processed_messages_batch(
        self,
        user_id: str | None = None,
        message_ids: List[str] | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> List[str]:
        """Check which messages have already been processed (batch mode)."""
        if not self.enable_facts or not user_id or not message_ids:
            return []

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            placeholders = ','.join('?' * len(message_ids))
            cursor.execute(f"""
                SELECT message_id FROM processed_messages
                WHERE user_id = ? AND message_id IN ({placeholders})
            """, [user_id] + message_ids)

            processed = [row[0] for row in cursor.fetchall()]

            if not self._persistent_conn:
                conn.close()

            return processed

        except Exception as e:
            logger.error(f"Failed to check processed messages batch: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return []

    def mark_processed_messages_batch(
        self,
        user_id: str | None = None,
        message_data: List[Dict[str, str]] | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """Mark multiple messages as processed (batch mode)."""
        if not self.enable_facts or not user_id or not message_data:
            return False

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            for msg in message_data:
                cursor.execute("""
                    INSERT OR IGNORE INTO processed_messages (user_id, message_id, thread_id)
                    VALUES (?, ?, ?)
                """, (user_id, msg.get("message_id"), msg.get("thread_id")))

            if self._persistent_conn:
                conn.commit()
            else:
                conn.commit()
                conn.close()

            return True

        except Exception as e:
            logger.error(f"Failed to mark processed messages batch: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return False

    # =========================================================================
    # Phase 3: Relevance Scoring Methods (Python-side calculations)
    # =========================================================================

    def calculate_relevance_score(
        self,
        fact_id: str,
        user_id: str,
        recency_weight: float = 0.4,
        access_weight: float = 0.3,
        usage_weight: float = 0.3
    ) -> float:
        """
        Calculate relevance score for a fact.

        Combines:
        - Recency (40%): Exponential decay over 365 days
        - Access frequency (30%): Normalized to 100 accesses
        - Usage rate (30%): Ratio of actual usage in responses

        Args:
            fact_id: Fact identifier
            user_id: User identifier
            recency_weight: Weight for recency score (default: 0.4)
            access_weight: Weight for access frequency (default: 0.3)
            usage_weight: Weight for usage rate (default: 0.3)

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not self.enable_facts:
            return 0.5

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get fact metadata
            cursor.execute("""
                SELECT created_at, access_count
                FROM facts
                WHERE id = ? AND user_id = ?
            """, (fact_id, user_id))

            row = cursor.fetchone()
            if not row:
                if not self._persistent_conn:
                    conn.close()
                return 0.5

            created_at_str, access_count = row

            # Calculate recency score (exponential decay)
            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            age_days = (datetime.now() - created_at).total_seconds() / 86400
            max_age_days = 365
            recency_score = math.exp(-age_days / max_age_days)

            # Calculate access score (normalized)
            max_access_count = 100
            access_score = min(access_count / max_access_count, 1.0)

            # Calculate usage score (from access log)
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN was_used = 1 THEN 1 END) as used_count
                FROM fact_access_log
                WHERE fact_id = ? AND was_used IS NOT NULL
            """, (fact_id,))

            log_row = cursor.fetchone()
            if log_row and log_row[0] > 0:
                usage_score = log_row[1] / log_row[0]
            else:
                usage_score = 0.5  # Default when no usage data

            if not self._persistent_conn:
                conn.close()

            # Combine scores
            final_score = (
                (recency_score * recency_weight) +
                (access_score * access_weight) +
                (usage_score * usage_weight)
            )

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            logger.error(f"Failed to calculate relevance score for {fact_id}: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return 0.5

    def update_fact_usage_feedback(
        self,
        access_log_ids: List[str],
        was_used: bool,
        credentials: Dict[str, Any] | None = None,
    ) -> int:
        """
        Update usage feedback for fact access log entries.

        Args:
            access_log_ids: List of access log IDs to update
            was_used: Whether facts were actually used in response
            credentials: Optional authentication credentials (ignored for SQLite)

        Returns:
            Number of records updated
        """
        if not self.enable_facts or not access_log_ids:
            return 0

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            placeholders = ','.join('?' * len(access_log_ids))
            cursor.execute(f"""
                UPDATE fact_access_log
                SET was_used = ?
                WHERE id IN ({placeholders})
            """, [1 if was_used else 0] + access_log_ids)

            updated_count = cursor.rowcount

            if self._persistent_conn:
                conn.commit()
            else:
                conn.commit()
                conn.close()

            return updated_count

        except Exception as e:
            logger.error(f"Failed to update fact usage feedback: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return 0

    def refresh_relevance_scores(
        self,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Recalculate relevance scores for all facts (or specific user).

        Args:
            user_id: Optional user ID to refresh scores for (None = all users)
            credentials: Optional authentication credentials (ignored for SQLite)

        Returns:
            Dictionary with statistics about updated scores
        """
        if not self.enable_facts:
            return {
                "updated_count": 0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0
            }

        conn: sqlite3.Connection | None = None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get facts to update
            if user_id:
                cursor.execute("""
                    SELECT id, user_id FROM facts WHERE user_id = ?
                """, (user_id,))
            else:
                cursor.execute("SELECT id, user_id FROM facts")

            facts = cursor.fetchall()

            scores = []
            for fact_id, fact_user_id in facts:
                score = self.calculate_relevance_score(fact_id, fact_user_id)
                cursor.execute("""
                    UPDATE facts
                    SET relevance_score = ?
                    WHERE id = ?
                """, (score, fact_id))
                scores.append(score)

            if self._persistent_conn:
                conn.commit()
            else:
                conn.commit()
                conn.close()

            if scores:
                return {
                    "updated_count": len(scores),
                    "avg_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores)
                }
            else:
                return {
                    "updated_count": 0,
                    "avg_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0
                }

        except Exception as e:
            logger.error(f"Failed to refresh relevance scores: {e}")
            if isinstance(conn, sqlite3.Connection) and not self._persistent_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return {
                "updated_count": 0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0
            }
