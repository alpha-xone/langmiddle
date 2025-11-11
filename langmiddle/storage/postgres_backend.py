"""
PostgreSQL storage backend implementation.

This module provides direct PostgreSQL implementation of the chat storage interface
using psycopg2 for database connections.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_graph_logger
from .base import SortOrder, ThreadSortBy
from .postgres_base import PostgreSQLBaseBackend

logger = get_graph_logger(__name__)

__all__ = ["PostgreSQLStorageBackend"]


class PostgreSQLStorageBackend(PostgreSQLBaseBackend):
    """Direct PostgreSQL implementation of chat storage backend."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        auto_create_tables: bool = False,
        enable_facts: bool = False,
        load_from_env: bool = True,
    ):
        """
        Initialize PostgreSQL storage backend.

        Args:
            connection_string: PostgreSQL connection string (optional if using .env)
            auto_create_tables: Whether to automatically create chat tables if they don't exist (default: False)
            enable_facts: Whether to create facts-related tables (requires auto_create_tables=True) (default: False)
            load_from_env: Whether to load connection string from .env file (default: True)

        Raises:
            ImportError: If psycopg2 dependencies are not installed
            ValueError: If connection string is not provided and not found in environment

        Example:
            # Using connection string directly (chat only)
            storage = ChatStorage.create(
                "postgres",
                connection_string="postgresql://user:password@localhost:5432/dbname",
                auto_create_tables=True
            )

            # With facts support
            storage = ChatStorage.create(
                "postgres",
                connection_string="postgresql://user:password@localhost:5432/dbname",
                auto_create_tables=True,
                enable_facts=True
            )

            # Using environment variables
            storage = ChatStorage.create("postgres")  # Loads from .env
        """
        # Try to import psycopg2
        try:
            from psycopg2 import pool  # noqa: F401
        except ImportError:
            raise ImportError(
                "psycopg2 dependencies not installed. "
                "Install with: pip install langmiddle[postgres] or pip install psycopg2-binary"
            )

        # Load from environment if requested
        if load_from_env and not connection_string:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                logger.debug("python-dotenv not installed, skipping .env file loading")

            connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

        # Validate connection string
        if not connection_string:
            raise ValueError(
                "PostgreSQL connection string not provided. Either:\n"
                "1. Pass connection_string parameter, or\n"
                "2. Set POSTGRES_CONNECTION_STRING environment variable, or\n"
                "3. Add it to a .env file in your project root\n\n"
                "Example: postgresql://user:password@localhost:5432/dbname"
            )

        self.connection_string = connection_string
        self._connection_pool = None

        # Initialize connection pool
        try:
            self._connection_pool = pool.SimpleConnectionPool(
                1, 10, connection_string  # min connections  # max connections
            )
            logger.debug("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise

        # Create tables if requested
        if auto_create_tables:
            sql_dir = Path(__file__).parent / "postgres"
            self._create_tables_with_psycopg2(
                connection_string, sql_dir, enable_facts=enable_facts
            )

    def _get_connection(self):
        """Get a connection from the pool."""
        if not self._connection_pool:
            raise RuntimeError("Connection pool not initialized")
        return self._connection_pool.getconn()

    def _return_connection(self, conn):
        """Return a connection to the pool."""
        if self._connection_pool:
            self._connection_pool.putconn(conn)

    def _execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
    ) -> Optional[Any]:
        """
        Execute a SQL query using psycopg2.

        Args:
            query: SQL query string
            params: Query parameters tuple
            fetch_one: Whether to fetch one result
            fetch_all: Whether to fetch all results

        Returns:
            Query results if fetch_one or fetch_all, None otherwise
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(query, params)

            if fetch_one:
                result = cursor.fetchone()
                cursor.close()
                conn.commit()
                return result
            elif fetch_all:
                results = cursor.fetchall()
                cursor.close()
                conn.commit()
                return results
            else:
                cursor.close()
                conn.commit()
                return None

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database query error: {e}")
            raise
        finally:
            if conn:
                self._return_connection(conn)

    def authenticate(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """
        Authenticate with PostgreSQL.

        For direct PostgreSQL, authentication is handled at connection time.
        This method is a no-op that always returns True.

        Args:
            credentials: Not used for PostgreSQL (authentication via connection string)

        Returns:
            True (authentication handled by connection string)
        """
        logger.debug("PostgreSQL authentication handled by connection string")
        return True

    def extract_user_id(self, credentials: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Extract user ID from credentials.

        For direct PostgreSQL without external auth system, user_id must be
        provided directly in credentials.

        Args:
            credentials: Dict containing 'user_id' key

        Returns:
            User ID if found, None otherwise
        """
        if not credentials:
            return None

        user_id = credentials.get("user_id")
        if user_id:
            return user_id

        logger.debug("No user_id found in credentials")
        return None

    def close(self):
        """Close the connection pool and clean up resources."""
        if self._connection_pool:
            self._connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

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
        try:
            result = self._execute_query(
                "SELECT id, user_id, created_at, updated_at, custom_state FROM chat_threads WHERE id = %s",
                params=(thread_id,),
                fetch_one=True,
            )
        except Exception as e:
            logger.error(f"Error retrieving thread record for ID {thread_id}: {e}")
            return None

        if not result:
            return None

        # Fetch messages for this thread
        try:
            messages = self._execute_query(
                "SELECT id, content, role, created_at, metadata, usage_metadata FROM chat_messages WHERE thread_id = %s ORDER BY created_at ASC",
                params=(thread_id,),
                fetch_all=True,
            ) or []
        except Exception as e:
            logger.error(f"Error retrieving messages for thread {thread_id}: {e}")
            messages = []

        msgs = []
        try:
            for row in messages:
                msg = {
                    "id": row[0],
                    "content": row[1],
                    "role": row[2],
                    "created_at": row[3],
                    "metadata": json.loads(row[4]) if isinstance(row[4], str) and row[4] else row[4],
                    "usage_metadata": json.loads(row[5]) if isinstance(row[5], str) and row[5] else row[5],
                }
                msgs.append(msg)
        except Exception:
            # If parsing fails, fall back to raw rows
            msgs = [
                {
                    "id": r[0],
                    "content": r[1],
                    "role": r[2],
                    "created_at": r[3],
                }
                for r in messages
            ]

        thread = {
            "thread_id": result[0],
            "user_id": result[1],
            "created_at": result[2],
            "updated_at": result[3],
            "metadata": result[4],
            "values": {"messages": msgs},
        }

        # Merge custom_state into values if present
        try:
            if result[4]:
                thread["values"].update(
                    result[4] if isinstance(result[4], dict) else json.loads(result[4])
                )
        except Exception:
            pass

        return thread

    def search_threads(
        self,
        *,
        metadata: dict | None = None,
        values: dict | None = None,
        ids: List[str] | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: ThreadSortBy | None = None,
        sort_order: SortOrder | None = None,
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
                placeholders = ", ".join(["%s"] * len(ids))
                conditions.append(f"id IN ({placeholders})")
                params.extend(ids)

            # Apply metadata filters (stored as JSON in custom_state)
            if metadata:
                for key, value in metadata.items():
                    conditions.append("custom_state->>%s = %s")
                    params.extend([key, str(value)])

            # Add WHERE clause if we have conditions
            if conditions:
                query_parts.append("WHERE " + " AND ".join(conditions))

            # Add sorting
            if sort_by:
                direction = "DESC" if sort_order == "desc" else "ASC"
                query_parts.append(f"ORDER BY {sort_by} {direction}")

            # Add limit and offset
            query_parts.append("LIMIT %s OFFSET %s")
            params.extend([limit, offset])

            query = " ".join(query_parts)

            results = self._execute_query(query, params=tuple(params), fetch_all=True)
        except Exception as e:
            logger.error(f"Error executing threads query: {e}")
            return []

        if not results:
            return []

        # Collect thread ids and fetch messages in one query
        thread_ids = [row[0] for row in results]
        msgs_by_thread = {}
        if thread_ids:
            try:
                placeholders = ", ".join(["%s"] * len(thread_ids))
                msg_query = f"SELECT id, content, role, created_at, metadata, usage_metadata, thread_id FROM chat_messages WHERE thread_id IN ({placeholders}) ORDER BY created_at ASC"
                msg_results = self._execute_query(msg_query, params=tuple(thread_ids), fetch_all=True) or []

                for row in msg_results:
                    tid = row[6]
                    m = {
                        "id": row[0],
                        "content": row[1],
                        "role": row[2],
                        "created_at": row[3],
                        "metadata": json.loads(row[4]) if isinstance(row[4], str) and row[4] else row[4],
                        "usage_metadata": json.loads(row[5]) if isinstance(row[5], str) and row[5] else row[5],
                    }
                    msgs_by_thread.setdefault(tid, []).append(m)
            except Exception as e:
                logger.error(f"Error retrieving messages for threads: {e}")

        threads = []

        for row in results:
            thread_id = row[0]
            thread_info = {
                "thread_id": thread_id,
                "user_id": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "metadata": row[4],
                "values": {"messages": msgs_by_thread.get(thread_id, [])},
            }
            try:
                if row[4]:
                    thread_info["values"].update(row[4] if isinstance(row[4], dict) else json.loads(row[4]))
            except Exception:
                pass
            threads.append(thread_info)

        logger.debug(f"Found {len(threads)} threads matching search criteria")
        return threads

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
        try:
            self._execute_query(
                "DELETE FROM chat_messages WHERE thread_id = %s",
                params=(thread_id,),
            )
        except Exception as e:
            logger.error(f"Error deleting messages for thread {thread_id}: {e}")
            return

        # Delete the thread record
        try:
            self._execute_query(
                "DELETE FROM chat_threads WHERE id = %s",
                params=(thread_id,),
            )
            logger.info(f"Deleted thread {thread_id} and all its messages")
        except Exception as e:
            logger.error(f"Error deleting thread {thread_id}: {e}")

    # Facts management methods (not yet implemented for direct PostgreSQL)
    def insert_fact(
        self,
        user_id: str,
        namespace: str,
        content: str,
        embedding: List[float],
    ) -> Optional[str]:
        """Insert a fact with its embedding."""
        raise NotImplementedError(
            "Facts management not yet implemented in PostgreSQL backend. "
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
        """Update a fact and its embedding."""
        raise NotImplementedError(
            "Facts management not yet implemented in PostgreSQL backend. "
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
        """Query facts using vector similarity."""
        raise NotImplementedError(
            "Facts management not yet implemented in PostgreSQL backend. "
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
            "Facts management not yet implemented in PostgreSQL backend. "
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
            "Facts management not yet implemented in PostgreSQL backend. "
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
            "Processed messages tracking not yet implemented in PostgreSQL backend. "
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
            "Processed messages tracking not yet implemented in PostgreSQL backend. "
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
            "Processed messages tracking not yet implemented in PostgreSQL backend. "
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
            "Processed messages tracking not yet implemented in PostgreSQL backend. "
            "Use SupabaseStorageBackend for facts support."
        )

    def get_or_create_embedding_table(
        self,
        dimension: int,
    ) -> bool:
        """Ensure an embedding table exists for the given dimension."""
        raise NotImplementedError(
            "Facts management not yet implemented in PostgreSQL backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )
