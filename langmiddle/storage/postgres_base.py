"""
PostgreSQL base backend implementation.

This module provides common PostgreSQL functionality that can be shared
between Supabase (which is PostgreSQL-based) and direct PostgreSQL backends.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
        self,
        connection_string: str,
        sql_dir: Path,
        enable_facts: bool = False,
        enable_day_dreaming: bool = False
    ) -> None:
        """
        Create PostgreSQL tables from SQL files if they don't exist.

        This method reads the SQL schema files and executes them to create the necessary tables.
        It's designed to be idempotent - safe to run multiple times with detailed logging.

        Args:
            connection_string: PostgreSQL connection string for direct database access
            sql_dir: Path to directory containing SQL schema files (e.g., 'postgres/' or 'supabase/')
            enable_facts: Whether to create facts-related tables (chat_facts.sql with processed_messages)
            enable_day_dreaming: Whether to create fact deduplication and maintenance functions
                                 (requires enable_facts=True)

        Raises:
            ImportError: If psycopg2 is not installed
            Exception: If table creation fails

        Note:
            - For PostgreSQL backend: uses generic SQL without authentication dependencies
            - For Supabase backend: uses Supabase-specific auth.users and RLS policies
            - All SQL scripts use IF NOT EXISTS/IF EXISTS checks for idempotency
            - Detailed logging reports on each table, index, trigger, function, and policy
            - fact_maintenance.sql provides server-side deduplication with pg_cron support
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

            logger.info(f"Starting table creation from SQL directory: {sql_dir}")

            # Connect to database
            conn = psycopg2.connect(connection_string)
            conn.autocommit = True
            cursor = conn.cursor()

            # Execute SQL files - let SQL handle idempotency with IF NOT EXISTS
            sql_files = ["chat_history.sql"]

            # Add facts-related SQL file if enabled (includes processed_messages table)
            if enable_facts:
                sql_files.append("chat_facts.sql")
                logger.info("Facts tables enabled - will create chat_facts.sql schema")

                # Add fact maintenance SQL if requested
                if enable_day_dreaming:
                    sql_files.append("fact_maintenance.sql")
                    logger.info("Fact maintenance enabled - will create deduplication functions and jobs")

            for sql_file in sql_files:
                sql_path = sql_dir / sql_file
                if not sql_path.exists():
                    logger.warning(f"SQL file not found: {sql_path}, skipping")
                    continue

                logger.info(f"Executing SQL file: {sql_file}")

                with open(sql_path, "r", encoding="utf-8") as f:
                    sql_content = f.read()

                # Split SQL into individual statements and execute with detailed logging
                statements = self._split_sql_statements(sql_content)

                for idx, statement in enumerate(statements, 1):
                    statement = statement.strip()
                    if not statement:
                        continue

                    stmt_name = ""
                    try:
                        # Determine statement type for logging
                        stmt_type = self._get_statement_type(statement)
                        stmt_name = self._extract_object_name(statement)

                        cursor.execute(statement)

                        # Log based on statement type
                        if stmt_type in [
                            'CREATE TABLE', 'CREATE INDEX', 'CREATE TRIGGER',
                            'CREATE FUNCTION', 'CREATE POLICY', 'ALTER TABLE'
                        ]:
                            logger.info(f"  [+] {stmt_type}: {stmt_name}")
                        elif stmt_type == 'DROP':
                            logger.debug(f"  [-] {stmt_type}: {stmt_name}")
                        elif stmt_type in ['COMMENT', 'DO']:
                            logger.debug(f"  [*] {stmt_type}")
                        else:
                            logger.debug(f"  [*] Executed statement {idx}")

                    except Exception as e:
                        # Log but don't fail on errors (many will be "already exists")
                        error_msg = str(e).lower()
                        if 'already exists' in error_msg or 'does not exist' in error_msg:
                            logger.debug(f"  [~] Skipped (already exists): {stmt_name}")
                        else:
                            logger.warning(f"  [!] Error in statement {idx}: {e}")

                logger.info(f"Completed processing {sql_file}")

            # Report final status of all tables
            self._log_table_status(cursor)

            cursor.close()
            conn.close()

            logger.info("Table creation process completed successfully")

        except Exception as e:
            logger.error(f"Failed to create PostgreSQL tables: {e}")
            raise Exception(
                f"Table creation failed: {e}\n\n"
                f"SQL files location: {sql_dir}"
            )

    def _split_sql_statements(self, sql_content: str) -> list:
        """
        Split SQL content into individual statements.

        Handles complex cases like function definitions with semicolons inside.
        """
        statements = []
        current = []
        in_function = False
        in_do_block = False

        for line in sql_content.split('\n'):
            stripped = line.strip()

            # Track function/do block boundaries
            if stripped.lower().startswith(('create function', 'create or replace function')):
                in_function = True
            elif stripped.lower().startswith('do $$'):
                in_do_block = True
            elif stripped == '$$;' and (in_function or in_do_block):
                current.append(line)
                statements.append('\n'.join(current))
                current = []
                in_function = False
                in_do_block = False
                continue

            current.append(line)

            # Split on semicolon only if not in function/do block
            if stripped.endswith(';') and not in_function and not in_do_block:
                statements.append('\n'.join(current))
                current = []

        if current:
            statements.append('\n'.join(current))

        return statements

    def _get_statement_type(self, statement: str) -> str:
        """Extract the type of SQL statement for logging."""
        stmt_lower = statement.lower().strip()

        if stmt_lower.startswith('create table'):
            return 'CREATE TABLE'
        elif stmt_lower.startswith('create index'):
            return 'CREATE INDEX'
        elif stmt_lower.startswith('create trigger'):
            return 'CREATE TRIGGER'
        elif stmt_lower.startswith('create or replace function'):
            return 'CREATE FUNCTION'
        elif stmt_lower.startswith('create function'):
            return 'CREATE FUNCTION'
        elif stmt_lower.startswith('create policy'):
            return 'CREATE POLICY'
        elif stmt_lower.startswith('create extension'):
            return 'CREATE EXTENSION'
        elif stmt_lower.startswith('alter table'):
            return 'ALTER TABLE'
        elif stmt_lower.startswith('drop'):
            return 'DROP'
        elif stmt_lower.startswith('comment'):
            return 'COMMENT'
        elif stmt_lower.startswith('do $$'):
            return 'DO'
        else:
            return 'STATEMENT'

    def _extract_object_name(self, statement: str) -> str:
        """Extract object name from SQL statement for logging."""
        stmt_lower = statement.lower().strip()

        try:
            # Handle various CREATE statements
            if 'create table' in stmt_lower:
                parts = stmt_lower.split('create table')[1].split()
                for part in parts:
                    if part not in ['if', 'not', 'exists']:
                        return part.strip('(').replace('public.', '')

            elif 'create index' in stmt_lower:
                parts = stmt_lower.split('create index')[1].split()
                for part in parts:
                    if part not in ['if', 'not', 'exists']:
                        return part.split()[0].replace('public.', '')

            elif 'create trigger' in stmt_lower:
                parts = stmt_lower.split('create trigger')[1].split()
                return parts[0].replace('public.', '')

            elif 'create function' in stmt_lower or 'create or replace function' in stmt_lower:
                if 'replace function' in stmt_lower:
                    parts = stmt_lower.split('function')[1].split('(')[0].strip()
                else:
                    parts = stmt_lower.split('create function')[1].split('(')[0].strip()
                return parts.replace('public.', '')

            elif 'create policy' in stmt_lower:
                parts = stmt_lower.split('create policy')[1].split('"')
                if len(parts) >= 2:
                    return parts[1]

            elif 'create extension' in stmt_lower:
                parts = stmt_lower.split('create extension')[1].split()
                for part in parts:
                    if part not in ['if', 'not', 'exists']:
                        return part.strip('"')

            elif 'alter table' in stmt_lower:
                parts = stmt_lower.split('alter table')[1].split()
                return parts[0].replace('public.', '')

            elif stmt_lower.startswith('drop'):
                words = stmt_lower.split()
                if len(words) >= 4 and words[0] == 'drop':
                    return words[4] if 'exists' in stmt_lower else words[2]

        except Exception:
            pass

        return "unknown"

    def _log_table_status(self, cursor) -> None:
        """Log the final status of all created tables."""
        try:
            # Check chat history tables
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('chat_threads', 'chat_messages')
                ORDER BY table_name
            """)
            history_tables = [row[0] for row in cursor.fetchall()]

            if history_tables:
                logger.info(f"Chat history tables present: {', '.join(history_tables)}")

            # Check facts tables
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('facts', 'processed_messages')
                ORDER BY table_name
            """)
            facts_tables = [row[0] for row in cursor.fetchall()]

            if facts_tables:
                logger.info(f"Facts tables present: {', '.join(facts_tables)}")

            # Check for dynamic embedding tables
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE 'fact_embeddings_%'
                ORDER BY table_name
            """)
            embedding_tables = [row[0] for row in cursor.fetchall()]

            if embedding_tables:
                logger.info(f"Embedding tables present: {', '.join(embedding_tables)}")

            # Check for functions
            cursor.execute("""
                SELECT routine_name
                FROM information_schema.routines
                WHERE routine_schema = 'public'
                AND routine_name IN ('update_updated_at_column', 'embedding_table_exists',
                                     'create_embedding_table', 'ensure_embedding_table', 'search_facts')
                ORDER BY routine_name
            """)
            functions = [row[0] for row in cursor.fetchall()]

            if functions:
                logger.info(f"Functions present: {', '.join(functions)}")

        except Exception as e:
            logger.debug(f"Could not check table status: {e}")

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

    def ensure_thread_exists(self, credentials: Dict[str, Any] | None, thread_id: str, user_id: str) -> bool:
        """
        Ensure chat thread exists in database.

        Args:
            credentials: Authentication credentials (unused for PostgreSQL)
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
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
        user_id: str,
        messages: List[AnyMessage],
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save messages to database.

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

        # Update custom_state in chat_threads if provided
        if custom_state:
            try:
                self._execute_query(
                    """
                    UPDATE chat_threads
                    SET custom_state = $1
                    WHERE thread_id = $2
                    """,
                    (custom_state, thread_id),
                )
            except Exception as e:
                logger.error(f"Error updating custom_state for thread {thread_id}: {e}")

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

    # =========================================================================
    # Facts Management Methods - Must be implemented by subclasses
    # =========================================================================

    def get_or_create_embedding_table(self, dimension: int) -> bool:
        """Ensure an embedding table exists for the given dimension."""
        raise NotImplementedError(
            "Facts management not implemented for this backend. "
            "Use SupabaseStorageBackend for facts support."
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
        """Insert facts with optional embeddings and cue embeddings into storage.

        Facts can be passed as simple strings for convenience.
        They will be automatically converted to fact dictionaries.

        Args:
            credentials: Authentication credentials
            user_id: User identifier
            facts: List of facts (strings or dicts)
            embeddings: Optional list of embedding vectors
            model_dimension: Dimension of embeddings
            cue_embeddings: Optional list of (cue_text, embedding) tuples per fact
        """
        raise NotImplementedError(
            "Facts management not implemented for this backend. "
            "Use SupabaseStorageBackend for facts support."
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
            "Facts management not implemented for this backend. "
            "Use SupabaseStorageBackend for facts support."
        )

    def get_fact_by_id(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a fact by its ID."""
        raise NotImplementedError(
            "Facts management not implemented for this backend. "
            "Use SupabaseStorageBackend for facts support."
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
            "Facts management not implemented for this backend. "
            "Use SupabaseStorageBackend for facts support."
        )

    def delete_fact(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete a fact and its embeddings."""
        raise NotImplementedError(
            "Facts management not implemented for this backend. "
            "Use SupabaseStorageBackend for facts support."
        )

    def get_fact_history(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get complete history for a specific fact."""
        raise NotImplementedError(
            "Fact history not implemented for this backend. "
            "Use SupabaseStorageBackend for fact history support."
        )

    def get_recent_fact_changes(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        limit: int = 50,
        operation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent fact changes for a user."""
        raise NotImplementedError(
            "Fact history not implemented for this backend. "
            "Use SupabaseStorageBackend for fact history support."
        )

    def get_fact_change_stats(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get statistics about fact changes for a user."""
        raise NotImplementedError(
            "Fact history not implemented for this backend. "
            "Use SupabaseStorageBackend for fact history support."
        )
