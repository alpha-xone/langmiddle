"""
PostgreSQL base backend implementation.

This module provides common PostgreSQL functionality that can be shared
between Supabase (which is PostgreSQL-based) and direct PostgreSQL backends.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import ChatStorageBackend

logger = get_graph_logger(__name__)

__all__ = ["PostgreSQLBaseBackend"]


class PostgreSQLBaseBackend(ChatStorageBackend):
    """
    Base class for PostgreSQL-based storage backends.

    Provides common functionality for direct PostgreSQL and Supabase backends.
    """

    def _create_tables_with_psycopg2(
        self, connection_string: str, sql_dir: Path
    ) -> None:
        """
        Create PostgreSQL tables from SQL files if they don't exist.

        This method reads the SQL schema files and executes them to create the necessary tables.
        It's designed to be idempotent - safe to run multiple times.

        Args:
            connection_string: PostgreSQL connection string for direct database access
            sql_dir: Path to directory containing SQL schema files

        Raises:
            ImportError: If psycopg2 is not installed
            Exception: If table creation fails
        """
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for automatic table creation. "
                "Install with: pip install psycopg2-binary"
            )

        try:
            if not sql_dir.exists():
                logger.error(f"SQL directory not found: {sql_dir}")
                raise FileNotFoundError(f"SQL schema files not found at {sql_dir}")

            # Connect to database
            conn = psycopg2.connect(connection_string)
            conn.autocommit = True
            cursor = conn.cursor()

            # Check if tables already exist
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'chat_threads'
                );
            """
            )
            result = cursor.fetchone()
            threads_exists = result[0] if result else False

            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'chat_messages'
                );
            """
            )
            result = cursor.fetchone()
            messages_exists = result[0] if result else False

            if threads_exists and messages_exists:
                logger.info("PostgreSQL tables already exist, skipping creation")
                cursor.close()
                conn.close()
                return

            logger.info("Creating PostgreSQL tables from SQL schema files...")

            # Create trigger function first (required by chat_threads)
            cursor.execute(
                """
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = timezone('utc'::text, now());
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """
            )
            logger.debug("Created update_updated_at_column trigger function")

            # Execute SQL files in order (threads first, then messages due to foreign key)
            sql_files = ["chat_threads.sql", "chat_messages.sql"]

            for sql_file in sql_files:
                sql_path = sql_dir / sql_file
                if not sql_path.exists():
                    logger.warning(f"SQL file not found: {sql_path}, skipping")
                    continue

                with open(sql_path, "r", encoding="utf-8") as f:
                    sql_content = f.read()

                cursor.execute(sql_content)
                logger.info(f"Successfully executed {sql_file}")

            cursor.close()
            conn.close()

            logger.info("Successfully created all PostgreSQL tables")

        except Exception as e:
            logger.error(f"Failed to create PostgreSQL tables: {e}")
            raise Exception(
                f"Table creation failed: {e}\n\n"
                f"SQL files location: {sql_dir}"
            )

    def _execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
    ) -> Optional[Any]:
        """
        Execute a SQL query using the backend's connection method.

        Must be implemented by subclasses to use their specific connection mechanism.

        Args:
            query: SQL query string
            params: Query parameters tuple
            fetch_one: Whether to fetch one result
            fetch_all: Whether to fetch all results

        Returns:
            Query results if fetch_one or fetch_all, None otherwise
        """
        raise NotImplementedError("Subclasses must implement _execute_query")

    def get_existing_message_ids(self, thread_id: str) -> set:
        """
        Get existing message IDs from database.

        Args:
            thread_id: Thread identifier

        Returns:
            Set of existing message IDs
        """
        try:
            results = self._execute_query(
                "SELECT id FROM chat_messages WHERE thread_id = %s",
                params=(thread_id,),
                fetch_all=True,
            )

            if results:
                message_ids = {row[0] for row in results}
                logger.debug(
                    f"Found {len(message_ids)} existing messages for thread {thread_id}"
                )
                return message_ids
            return set()

        except Exception as e:
            logger.error(f"Error fetching existing messages: {e}")
            return set()

    def ensure_thread_exists(self, thread_id: str, user_id: str) -> bool:
        """
        Ensure chat thread exists in database.

        Args:
            thread_id: Thread identifier
            user_id: User identifier

        Returns:
            True if thread exists or was created
        """
        try:
            self._execute_query(
                """
                INSERT INTO chat_threads (id, user_id)
                VALUES (%s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                params=(thread_id, user_id),
            )
            logger.debug(f"Chat thread {thread_id} ensured in database")
            return True

        except Exception as e:
            logger.error(f"Error upserting chat thread: {e}")
            return False

    def save_messages(
        self, thread_id: str, user_id: str, messages: List[AnyMessage]
    ) -> Dict[str, Any]:
        """
        Save messages to database.

        Args:
            thread_id: Thread identifier
            user_id: User identifier
            messages: List of messages to save

        Returns:
            Dict with 'saved_count' and 'errors' keys
        """
        saved_count = 0
        errors = []

        for msg in messages:
            try:
                # Prepare message data
                role = self.TYPE_TO_ROLE.get(msg.type, msg.type)
                content = msg.content
                metadata = getattr(msg, "response_metadata", {})
                usage_metadata = getattr(msg, "usage_metadata", {})

                # Convert metadata to JSON string for psycopg2
                import json

                metadata_json = json.dumps(metadata) if metadata else "{}"
                usage_metadata_json = (
                    json.dumps(usage_metadata) if usage_metadata else None
                )

                # Save to database
                self._execute_query(
                    """
                    INSERT INTO chat_messages (id, user_id, thread_id, content, role, metadata, usage_metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        role = EXCLUDED.role,
                        metadata = EXCLUDED.metadata,
                        usage_metadata = EXCLUDED.usage_metadata
                    """,
                    params=(
                        msg.id,
                        user_id,
                        thread_id,
                        content,
                        role,
                        metadata_json,
                        usage_metadata_json,
                    ),
                )

                time.sleep(0.05)  # Small delay to avoid potential rate limiting

                saved_count += 1
                logger.debug(f"Saved message {msg.id} to database")

            except Exception as e:
                errors.append(f"Error saving message {msg.id}: {e}")
                logger.error(f"Error saving message {msg.id}: {e}")

        return {"saved_count": saved_count, "errors": errors}
