"""
Supabase storage backend implementation.

This module provides Supabase-specific implementation of the chat storage interface.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from jose import JWTError, jwt
from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import ChatStorageBackend

logger = get_graph_logger(__name__)

__all__ = ["SupabaseStorageBackend"]


class SupabaseStorageBackend(ChatStorageBackend):
    """Supabase implementation of chat storage backend."""

    def __init__(
        self,
        client=None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        connection_string: Optional[str] = None,
        auto_create_tables: bool = False,
        load_from_env: bool = True,
    ):
        """
        Initialize Supabase storage backend.

        Args:
            client: Optional Supabase client instance (if provided, other params are ignored)
            supabase_url: Supabase project URL (optional if client provided or using .env)
            supabase_key: Supabase anonymous key (optional if client provided or using .env)
            connection_string: PostgreSQL connection string for table creation (optional, only needed if auto_create_tables=True)
            auto_create_tables: Whether to automatically create tables if they don't exist (default: False)
            load_from_env: Whether to load credentials from .env file (default: True)

        Raises:
            ImportError: If Supabase dependencies are not installed
            ValueError: If credentials are not provided and not found in environment

        Note:
            For first-time setup with auto_create_tables=True, you need to provide connection_string.
            This is only needed once to create the tables. After that, use normal supabase_url/supabase_key.

            Example first-time setup:
                storage = ChatStorage.create(
                    "supabase",
                    supabase_url="https://your-project.supabase.co",
                    supabase_key="your-anon-key",
                    connection_string="postgresql://postgres:password@db.xxx.supabase.co:5432/postgres",
                    auto_create_tables=True
                )

            Example normal usage (after tables exist):
                storage = ChatStorage.create("supabase")  # Loads from .env
        """
        if client:
            self.client = client
            self._authenticated = False
            return

        # Try to import Supabase
        try:
            from supabase import create_client
        except ImportError:
            raise ImportError(
                "Supabase dependencies not installed. "
                "Install with: pip install langmiddle[supabase]"
            )

        # Load from environment if requested
        if load_from_env and (not supabase_url or not supabase_key):
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                logger.debug("python-dotenv not installed, skipping .env file loading")

            supabase_url = supabase_url or os.getenv("SUPABASE_URL")
            supabase_key = supabase_key or os.getenv("SUPABASE_ANON_KEY")
            if not connection_string:
                connection_string = os.getenv("SUPABASE_CONNECTION_STRING")

        # Validate credentials
        if not supabase_url or not supabase_key:
            raise ValueError(
                "Supabase credentials not provided. Either:\n"
                "1. Pass supabase_url and supabase_key parameters, or\n"
                "2. Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables, or\n"
                "3. Add them to a .env file in your project root"
            )

        # Create Supabase client
        try:
            self.client = create_client(supabase_url, supabase_key)
            self._authenticated = False
            logger.debug("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

        # Create tables if requested
        if auto_create_tables:
            if not connection_string:
                raise ValueError(
                    "connection_string is required when auto_create_tables=True. "
                    "Get it from your Supabase project settings under Database > Connection string (Direct connection)."
                )
            self._create_tables_if_needed(connection_string)

    def _create_tables_if_needed(self, connection_string: str) -> None:
        """
        Create Supabase tables from SQL files if they don't exist.

        This method reads the SQL schema files and executes them to create the necessary tables.
        It's designed to be idempotent - safe to run multiple times.

        Args:
            connection_string: PostgreSQL connection string for direct database access

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
            # Get the path to SQL files
            sql_dir = Path(__file__).parent / "supabase"

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
                logger.info("Supabase tables already exist, skipping creation")
                cursor.close()
                conn.close()
                return

            logger.info("Creating Supabase tables from SQL schema files...")

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

            logger.info("Successfully created all Supabase tables")

        except Exception as e:
            logger.error(f"Failed to create Supabase tables: {e}")
            raise Exception(
                f"Table creation failed: {e}\n\n"
                "You can also create tables manually in the Supabase SQL Editor:\n"
                f"1. Go to your Supabase project dashboard\n"
                f"2. Navigate to SQL Editor\n"
                f"3. Run the SQL files from: {Path(__file__).parent / 'supabase'}"
            )

    def authenticate(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """
        Authenticate with Supabase using JWT.

        Args:
            credentials: Dict containing 'jwt_token' key

        Returns:
            True if authentication successful or not required
        """
        jwt_token = credentials.get("jwt_token") if credentials else None
        if not jwt_token:
            logger.debug("No JWT token provided, allowing non-RLS access")
            return True  # Allow non-RLS access

        try:
            self.client.postgrest.auth(jwt_token)
            self._authenticated = True
            logger.debug("Successfully authenticated with Supabase")
            return True
        except Exception as e:
            logger.error(f"Supabase authentication failed: {e}")
            return False

    def extract_user_id(self, credentials: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Extract user ID from JWT token or direct user_id.

        Args:
            credentials: Dict containing 'jwt_token' and/or 'user_id'

        Returns:
            User ID if found, None otherwise
        """
        # Check for direct user_id first
        user_id = credentials.get("user_id") if credentials else None
        if user_id:
            return user_id

        # Extract from JWT token
        jwt_token = credentials.get("jwt_token") if credentials else None
        if not jwt_token:
            return None

        try:
            payload = jwt.get_unverified_claims(jwt_token)
            return payload.get("sub")
        except JWTError as e:
            logger.error(f"Error decoding JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting user_id from JWT: {e}")
            return None

    def get_existing_message_ids(self, thread_id: str) -> set:
        """
        Get existing message IDs from Supabase.

        Args:
            thread_id: Thread identifier

        Returns:
            Set of existing message IDs
        """
        try:
            result = (
                self.client.table("chat_messages")
                .select("id")
                .eq("thread_id", thread_id)
                .execute()
            )

            if result.data:
                message_ids = {
                    msg["id"]
                    for msg in result.data
                    if isinstance(msg, dict) and "id" in msg
                }
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
        Ensure chat thread exists in Supabase.

        Args:
            thread_id: Thread identifier
            user_id: User identifier

        Returns:
            True if thread exists or was created
        """
        try:
            result = (
                self.client.table("chat_threads")
                .upsert(
                    {
                        "id": thread_id,
                        "user_id": user_id,
                    },
                    on_conflict="id",
                )
                .execute()
            )

            if not result.data:
                logger.warning(
                    f"Chat thread upsert returned no data for thread {thread_id}"
                )
                return False
            else:
                logger.debug(f"Chat thread {thread_id} ensured in database")
                return True

        except Exception as e:
            logger.error(f"Error upserting chat thread: {e}")
            return False

    def save_messages(
        self, thread_id: str, user_id: str, messages: List[AnyMessage]
    ) -> Dict[str, Any]:
        """
        Save messages to Supabase.

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
                msg_data = {
                    "id": msg.id,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "content": msg.content,
                    "role": self.TYPE_TO_ROLE.get(msg.type, msg.type),
                    "metadata": getattr(msg, "response_metadata", {}),
                    "usage_metadata": getattr(msg, "usage_metadata", {}),
                }

                # Save to database
                result = (
                    self.client.table("chat_messages")
                    .upsert(msg_data, on_conflict="id")
                    .execute()
                )

                time.sleep(0.05)  # Small delay to avoid rate limiting

                if not result.data:
                    errors.append(f"Failed to save message {msg.id}")
                    logger.error(f"Failed to save message {msg.id}")
                else:
                    saved_count += 1
                    logger.debug(f"Saved message {msg.id} to database")

            except Exception as e:
                errors.append(f"Error saving message {msg.id}: {e}")
                logger.error(f"Error saving message {msg.id}: {e}")

        return {"saved_count": saved_count, "errors": errors}
