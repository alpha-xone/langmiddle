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
from .base import SortOrder, ThreadSortBy
from .postgres_base import PostgreSQLBaseBackend

logger = get_graph_logger(__name__)

__all__ = ["SupabaseStorageBackend"]


def thread_to_dict(thread: dict, messages: List[dict]) -> dict:
    """
    Convert a Supabase thread record to a dictionary.

    Args:
        thread: Supabase thread record
        messages: List of messages associated with the thread

    Returns:
        dict representation of the thread
    """
    thread_id = thread.get("id")
    data = {
        "thread_id": thread_id,
        "title": thread.get("title"),
        "created_at": thread.get("created_at"),
        "updated_at": thread.get("updated_at"),
        "metadata": thread.get("metadata"),
        "values": {
            "messages": [
                {
                    "content": msg.get("content"),
                    "role": msg.get("role"),
                    "created_at": msg.get("created_at"),
                    "metadata": msg.get("metadata"),
                    "usage_metadata": msg.get("usage_metadata"),
                    "id": msg.get("id"),
                }
                for msg in messages
                if msg.get("thread_id") == thread_id
            ],
        },
    }
    if thread.get("custom_state"):
        data["values"].update(thread["custom_state"])

    return data


class SupabaseStorageBackend(PostgreSQLBaseBackend):
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
            sql_dir = Path(__file__).parent / "supabase"
            self._create_tables_with_psycopg2(connection_string, sql_dir)

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

    def _execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
    ) -> Optional[Any]:
        """
        Execute a SQL query using Supabase's query builder.

        This wraps the parent class's SQL-based approach with Supabase's REST API.
        Note: For save_messages and other operations, we override to use Supabase client directly.

        Args:
            query: SQL query string (converted to Supabase operations)
            params: Query parameters tuple
            fetch_one: Whether to fetch one result
            fetch_all: Whether to fetch all results

        Returns:
            Query results if fetch_one or fetch_all, None otherwise
        """
        # For simple SELECT queries, we can use Supabase client
        # This is a simplified implementation - for complex queries,
        # you might want to use a direct PostgreSQL connection
        raise NotImplementedError(
            "Direct SQL execution not supported via Supabase client. "
            "Use Supabase query builder methods instead."
        )

    def get_existing_message_ids(self, thread_id: str) -> set:
        """
        Get existing message IDs from Supabase.

        Args:
            thread_id: Thread identifier

        Returns:
            Set of existing message IDs
        """
        # Execute query separately and handle errors per operation
        try:
            result = (
                self.client.table("chat_messages")
                .select("id")
                .eq("thread_id", thread_id)
                .execute()
            )
        except Exception as e:
            logger.error(f"Error executing message id query for thread {thread_id}: {e}")
            return set()

        try:
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
            logger.error(f"Error processing message id results for thread {thread_id}: {e}")
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
        self,
        thread_id: str,
        user_id: str,
        messages: List[AnyMessage],
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save messages to Supabase.

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

        if not self.ensure_thread_exists(thread_id, user_id):
            errors.append(f"Failed to ensure thread {thread_id} exists")
            return {"saved_count": saved_count, "errors": errors}

        # Update custom_state in chat_threads if provided
        if custom_state:
            try:
                self.client.table("chat_threads").update(
                    {"custom_state": custom_state}
                ).eq("id", thread_id).execute()
            except Exception as e:
                errors.append(f"Failed to update custom_state for thread {thread_id}: {e}")

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

                time.sleep(0.05)  # Small delay to avoid duplicated time

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

    def get_thread(
        self,
        thread_id: str,
    ) -> dict | None:
        """
        Get a thread by ID.

        Args:
            thread_id: The ID of the thread to get.
        """
        # Fetch thread record
        try:
            thread = (
                self.client
                .table("chat_threads")
                .select("*")
                .eq("id", thread_id)
                .execute()
            )
        except Exception as e:
            logger.error(f"Error executing thread query for id {thread_id}: {e}")
            return None

        if not thread.data:
            return None

        # Fetch messages for the thread in a separate operation
        try:
            messages = (
                self.client
                .table("chat_messages")
                .select("*")
                .eq("thread_id", thread_id)
                .order("created_at", desc=False)
                .execute()
            )
            msgs = messages.data if messages.data else []
        except Exception as e:
            logger.error(f"Error executing messages query for thread {thread_id}: {e}")
            msgs = []

        try:
            return thread_to_dict(thread.data[0], msgs)
        except Exception as e:
            logger.error(f"Error building thread dict for id {thread_id}: {e}")
            return None

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
            headers: Optional custom headers to include with the request.

        Returns:
            list[dict]: List of the threads matching the search parameters.
        """
        # Build and execute threads query
        try:
            query = (
                self.client
                .table("chat_threads")
                .select("*")
            )

            if ids:
                query = query.in_("id", ids)

            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    query = query.filter(f"metadata->>{key}", "eq", value)

            threads = (
                query
                .order("created_at", desc=True if sort_order is None else sort_order == "desc")
                .offset(size=offset)
                .limit(size=limit)
                .execute()
            )
        except Exception as e:
            logger.error(f"Error executing threads query: {e}")
            return []

        if not threads.data:
            return []

        logger.debug(f"Found {len(threads.data)} threads matching search criteria")

        # Fetch messages for all returned threads in a separate operation
        thread_ids = [thread["id"] for thread in threads.data]
        msgs = []
        try:
            messages = (
                self.client
                .table("chat_messages")
                .select("*")
                .in_("thread_id", thread_ids)
                .order("created_at", desc=False)
                .execute()
            )
            msgs = messages.data if messages.data else []
        except Exception as e:
            logger.error(f"Error executing messages query for threads: {e}")
            # Return threads without attached messages
            return threads.data

        logger.debug(f"Found {len(msgs)} messages matching search criteria")

        try:
            return [thread_to_dict(thread, msgs) for thread in threads.data]
        except Exception as e:
            logger.error(f"Error building thread dicts from results: {e}")
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
        try:
            result = (
                self.client
                .table("chat_threads")
                .delete()
                .eq("id", thread_id)
                .execute()
            )
            logger.info(str(result.data))

        except Exception as e:
            logger.error(f"Error retrieving threads: {e}")
