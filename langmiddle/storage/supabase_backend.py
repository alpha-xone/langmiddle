"""
Supabase storage backend implementation.

This module provides Supabase-specific implementation of the chat storage interface.
"""

import functools
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from jose import JWTError, jwt
from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import SortOrder, ThreadSortBy
from .postgres_base import PostgreSQLBaseBackend

logger = get_graph_logger(__name__)

__all__ = ["SupabaseStorageBackend"]


def require_auth(func: Callable) -> Callable:
    """
    Decorator to handle authentication for Supabase operations.

    This decorator:
    1. Extracts credentials from method arguments
    2. Authenticates with Supabase before operation (stateless)
    3. Logs authentication failures
    4. Returns error result if authentication fails

    Note: This decorator does NOT mutate instance state to ensure thread safety.

    Args:
        func: Method to wrap with authentication

    Returns:
        Wrapped method that handles authentication
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract credentials - look for 'credentials' in kwargs or as first positional arg
        credentials = kwargs.get('credentials')
        if credentials is None and args:
            # Check if first arg looks like credentials dict
            if isinstance(args[0], dict) and ('jwt_token' in args[0] or 'user_id' in args[0]):
                credentials = args[0]

        # Authenticate (stateless - only sets JWT on client, no instance mutation)
        if credentials:
            auth_success = self._authenticate_request(credentials)
            if not auth_success:
                logger.error(f"Authentication failed for {func.__name__}")
                # Return appropriate error based on return type
                return_annotation = func.__annotations__.get('return', None)
                if return_annotation == bool:
                    return False
                elif return_annotation == dict or 'Dict' in str(return_annotation):
                    return {"success": False, "error": "Authentication failed"}
                elif return_annotation == list or 'List' in str(return_annotation):
                    return []
                else:
                    return None

        # Execute the actual method
        return func(self, *args, **kwargs)

    return wrapper


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
        enable_facts: bool = False,
        load_from_env: bool = True,
        user_id: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Supabase storage backend.

        Args:
            client: Optional Supabase client instance (if provided, other params are ignored)
            supabase_url: Supabase project URL (optional if client provided or using .env)
            supabase_key: Supabase anonymous key (optional if client provided or using .env)
            connection_string: PostgreSQL connection string for table creation (optional, only needed if auto_create_tables=True)
            auto_create_tables: Whether to automatically create chat tables if they don't exist (default: False)
            enable_facts: Whether to create facts-related tables (requires auto_create_tables=True) (default: False)
            load_from_env: Whether to load credentials from .env file (default: True)
            user_id: Optional user_id to store at instance level (can also be set later via set_user)
            credentials: Optional credentials to authenticate on init (can also be set later via set_user)

        Raises:
            ImportError: If Supabase dependencies are not installed
            ValueError: If credentials are not provided and not found in environment

        Note:
            For first-time setup with auto_create_tables=True, you need to provide connection_string.
            This is only needed once to create the tables. After that, use normal supabase_url/supabase_key.

            Example first-time setup with chat only:
                storage = ChatStorage.create(
                    "supabase",
                    supabase_url="https://your-project.supabase.co",
                    supabase_key="your-anon-key",
                    connection_string="postgresql://postgres:password@db.xxx.supabase.co:5432/postgres",
                    auto_create_tables=True,
                )

            Example first-time setup with facts enabled:
                storage = ChatStorage.create(
                    "supabase",
                    supabase_url="https://your-project.supabase.co",
                    supabase_key="your-anon-key",
                    connection_string="postgresql://postgres:password@db.xxx.supabase.co:5432/postgres",
                    auto_create_tables=True,
                    enable_facts=True,
                )

            Example normal usage (after tables exist):
                storage = ChatStorage.create("supabase")  # Loads from .env

            Example with user context:
                storage = ChatStorage.create(
                    "supabase",
                    user_id="user123",
                    credentials={"jwt_token": "eyJhbGc..."}
                )
        """
        if client:
            self.client = client
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
            self._create_tables_with_psycopg2(
                connection_string=connection_string,
                sql_dir=sql_dir,
                enable_facts=enable_facts,
            )

    def prepare_credentials(
        self,
        user_id: str,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare Supabase-specific credentials.

        Args:
            user_id: User identifier
            auth_token: JWT token for RLS authentication

        Returns:
            Dict with 'user_id' and 'jwt_token' keys
        """
        credentials = {"user_id": user_id}
        if auth_token:
            credentials["jwt_token"] = auth_token
        return credentials

    def _authenticate_request(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """
        Authenticate a single request with Supabase using JWT.

        THREAD-SAFE: Does not mutate instance state, only sets JWT on client.
        Each request authenticates independently.

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
            # Set JWT on client - Supabase validates on each request
            self.client.postgrest.auth(jwt_token)
            logger.debug("JWT token set on Supabase client")
            return True
        except Exception as e:
            logger.error(f"Failed to set JWT on Supabase client: {e}")
            return False

    def authenticate(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """
        Authenticate with Supabase using JWT.

        Wrapper for backward compatibility. Use _authenticate_request internally.

        Args:
            credentials: Dict containing 'jwt_token' key

        Returns:
            True if authentication successful or not required
        """
        return self._authenticate_request(credentials)

    def extract_user_id(self, credentials: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Extract user ID from credentials.

        THREAD-SAFE: Pure function, no instance state mutation.

        Priority:
        1. Direct 'user_id' in credentials
        2. Extract from JWT 'sub' claim

        Args:
            credentials: Dict containing 'jwt_token' and/or 'user_id'

        Returns:
            User ID if found, None otherwise
        """
        if not credentials:
            return None

        # 1. Direct user_id in credentials
        user_id = credentials.get("user_id")
        if user_id:
            return user_id

        # 2. Extract from JWT token
        jwt_token = credentials.get("jwt_token")
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

    @require_auth
    def save_messages(
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
        messages: List[AnyMessage],
        user_id: Optional[str] = None,
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save messages to Supabase.

        Args:
            credentials: Authentication credentials
            thread_id: Thread identifier
            messages: List of messages to save
            user_id: Optional user ID (extracted from credentials if not provided)
            custom_state: Optional custom state defined in the graph

        Returns:
            Dict with 'saved_count' and 'errors' keys
        """
        # Extract user_id from credentials if not provided
        user_id = user_id or self.extract_user_id(credentials)
        if not user_id:
            return {"saved_count": 0, "errors": ["user_id required"]}

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
                ).eq("id", thread_id).eq("user_id", user_id).execute()
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

                # Save to database - let DB handle timestamps
                result = (
                    self.client.table("chat_messages")
                    .upsert(msg_data, on_conflict="id")
                    .execute()
                )

                # Small delay to ensure timestamps are different when messages arrive rapidly
                time.sleep(0.01)

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

    @require_auth
    def get_thread(
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
        user_id: Optional[str] = None,
    ) -> dict | None:
        """
        Get a thread by ID.

        Args:
            credentials: Authentication credentials
            thread_id: The ID of the thread to get.
            user_id: Optional user ID (extracted from credentials if not provided)
        """
        # Extract user_id from credentials if not provided
        user_id = user_id or self.extract_user_id(credentials)
        if not user_id:
            logger.error("user_id required for get_thread")
            return None

        # Fetch thread record - explicit tenant filtering
        try:
            thread = (
                self.client
                .table("chat_threads")
                .select("*")
                .eq("id", thread_id)
                .eq("user_id", user_id)  # Defense in depth: explicit tenant filter
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
                .eq("user_id", user_id)  # Defense in depth: explicit tenant filter
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

    @require_auth
    def search_threads(
        self,
        credentials: Optional[Dict[str, Any]],
        *,
        user_id: Optional[str] = None,
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
            credentials: Authentication credentials
            user_id: Optional user ID (extracted from credentials if not provided)
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
        # Extract user_id from credentials if not provided
        user_id = user_id or self.extract_user_id(credentials)
        if not user_id:
            logger.error("user_id required for search_threads")
            return []

        # Build and execute threads query
        try:
            query = (
                self.client
                .table("chat_threads")
                .select("*")
                .eq("user_id", user_id)  # Defense in depth: explicit tenant filter
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
                .eq("user_id", user_id)  # Defense in depth: explicit tenant filter
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

    @require_auth
    def delete_thread(
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
        user_id: Optional[str] = None,
    ):
        """
        Delete a thread.

        Args:
            credentials: Authentication credentials
            thread_id: The ID of the thread to delete.
            user_id: Optional user ID (extracted from credentials if not provided)

        Returns:
            None
        """
        # Extract user_id from credentials if not provided
        user_id = user_id or self.extract_user_id(credentials)
        if not user_id:
            logger.error("user_id required for delete_thread")
            return

        try:
            result = (
                self.client
                .table("chat_threads")
                .delete()
                .eq("id", thread_id)
                .eq("user_id", user_id)  # Defense in depth: explicit tenant filter
                .execute()
            )
            logger.info(str(result.data))

        except Exception as e:
            logger.error(f"Error deleting thread: {e}")

    # =========================================================================
    # Facts Management Methods
    # =========================================================================

    def get_or_create_embedding_table(
        self,
        dimension: int,
    ) -> bool:
        """
        Ensure an embedding table exists for the given dimension.

        Args:
            dimension: Embedding vector dimension

        Returns:
            True if table exists or was created, False otherwise
        """
        try:
            self.client.rpc(
                "ensure_embedding_table",
                {"p_dimension": dimension}
            ).execute()

            logger.debug(f"Embedding table for dimension {dimension} is ready")
            return True
        except Exception as e:
            logger.error(f"Error creating/checking embedding table for dimension {dimension}: {e}")
            return False

    @require_auth
    def insert_facts(
        self,
        credentials: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        facts: Optional[Sequence[Dict[str, Any] | str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        model_dimension: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Insert facts with optional embeddings into storage.

        Args:
            credentials: Optional authentication credentials (not needed if set_user was called)
            user_id: Optional user identifier (uses instance-level user_id if not provided)
            facts: List of facts. Each fact can be either:
                - A string (auto-converted to fact dictionary)
                - A dictionary with keys: content, namespace, language, intensity, confidence
            embeddings: Optional list of embedding vectors (must match length of facts)
            model_dimension: Dimension of the embedding vectors (inferred if not provided)

        Returns:
            Dict with 'inserted_count', 'fact_ids', and 'errors' keys
        """
        inserted_count = 0
        fact_ids = []
        errors = []

        if not facts:
            return {"inserted_count": 0, "fact_ids": [], "errors": ["No facts provided"]}

        # Normalize facts: convert strings to dictionaries
        normalized_facts: List[Dict[str, Any]] = []
        for idx, fact in enumerate(facts):
            if isinstance(fact, str):
                # Auto-convert string to fact dictionary
                normalized_facts.append({
                    "content": fact,
                    "namespace": [],
                    "language": "en",
                })
                logger.debug(f"Auto-converted string fact at index {idx} to dictionary")
            elif isinstance(fact, dict):
                normalized_facts.append(fact)
            else:
                return {
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": [
                        f"Fact at index {idx} must be a string or dictionary, got {type(fact).__name__}"
                    ]
                }

        # After normalization, all facts are guaranteed to be dictionaries
        facts_dicts: List[Dict[str, Any]] = normalized_facts

        # Validate embeddings if provided
        if embeddings:
            if len(embeddings) != len(facts_dicts):
                return {
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": [f"Embeddings count ({len(embeddings)}) must match facts count ({len(facts_dicts)})"]
                }

            if not model_dimension:
                model_dimension = len(embeddings[0]) if embeddings else None

            if model_dimension:
                # Ensure embedding table exists
                if not self.get_or_create_embedding_table(model_dimension):
                    return {
                        "inserted_count": 0,
                        "fact_ids": [],
                        "errors": [f"Failed to create/verify embedding table for dimension {model_dimension}"]
                    }

        # Insert facts
        for idx, fact in enumerate(facts_dicts):
            try:
                # Prepare fact data
                fact_data = {
                    "user_id": user_id,
                    "content": fact.get("content"),
                    "namespace": fact.get("namespace", []),
                    "language": fact.get("language", "en"),
                    "intensity": fact.get("intensity"),
                    "confidence": fact.get("confidence"),
                    "model_dimension": model_dimension,
                }

                # Insert into facts table
                result = self.client.table("facts").insert(fact_data).execute()

                if not result.data:
                    errors.append(f"Failed to insert fact at index {idx}: No data returned")
                    continue

                fact_id = result.data[0]["id"]
                fact_ids.append(fact_id)
                inserted_count += 1

                # Insert embedding if provided
                if embeddings and idx < len(embeddings):
                    try:
                        embedding_data = {
                            "fact_id": fact_id,
                            "embedding": embeddings[idx],
                        }

                        table_name = f"fact_embeddings_{model_dimension}"
                        emb_result = self.client.table(table_name).insert(embedding_data).execute()

                        if not emb_result.data:
                            logger.warning(f"Failed to insert embedding for fact {fact_id}")
                            errors.append(f"Inserted fact {fact_id} but failed to insert embedding")
                    except Exception as e:
                        logger.error(f"Error inserting embedding for fact {fact_id}: {e}")
                        errors.append(f"Inserted fact {fact_id} but error inserting embedding: {str(e)}")

            except Exception as e:
                logger.error(f"Error inserting fact at index {idx}: {e}")
                errors.append(f"Error inserting fact at index {idx}: {str(e)}")

        logger.info(f"Inserted {inserted_count} facts with {len(errors)} errors")
        return {
            "inserted_count": inserted_count,
            "fact_ids": fact_ids,
            "errors": errors
        }

    @require_auth
    def query_facts(
        self,
        credentials: Optional[Dict[str, Any]],
        query_embedding: List[float],
        user_id: str,
        model_dimension: int,
        match_threshold: float = 0.75,
        match_count: int = 10,
        filter_namespaces: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query facts using vector similarity search.

        Args:
            credentials: Authentication credentials
            query_embedding: Query vector for similarity search
            user_id: User identifier for filtering
            model_dimension: Dimension of the embedding model
            match_threshold: Minimum similarity threshold (0-1)
            match_count: Maximum number of results to return
            filter_namespaces: Optional list of namespace paths to filter by

        Returns:
            List of fact dictionaries with similarity scores
        """
        try:
            # Convert empty list to None to avoid PostgreSQL array dimension errors
            namespaces_param = None
            if filter_namespaces:
                namespaces_param = filter_namespaces

            # Prepare parameters for RPC call
            params = {
                "p_embedding": query_embedding,
                "p_dimension": model_dimension,
                "p_user_id": user_id,
                "p_threshold": match_threshold,
                "p_limit": match_count,
                "p_namespaces": namespaces_param,
            }

            # Call the search function
            result = self.client.rpc("search_facts", params).execute()

            if not result.data:
                logger.debug(f"No facts found matching query with threshold {match_threshold}")
                return []

            logger.info(f"Found {len(result.data)} facts matching query")
            return result.data

        except Exception as e:
            logger.error(f"Error querying facts: {e}")
            return []

    @require_auth
    def get_fact_by_id(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a fact by its ID.

        Args:
            credentials: Authentication credentials
            fact_id: Fact identifier
            user_id: User identifier for authorization

        Returns:
            Fact dictionary if found, None otherwise
        """
        try:
            result = (
                self.client.table("facts")
                .select("*")
                .eq("id", fact_id)
                .eq("user_id", user_id)
                .execute()
            )

            if not result.data:
                logger.debug(f"Fact {fact_id} not found for user {user_id}")
                return None

            return result.data[0]

        except Exception as e:
            logger.error(f"Error getting fact {fact_id}: {e}")
            return None

    @require_auth
    def update_fact(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: str,
        updates: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Update a fact's content and/or metadata.

        Args:
            credentials: Authentication credentials
            fact_id: Fact identifier
            user_id: User identifier for authorization
            updates: Dictionary of fields to update (content, namespace, intensity, confidence, etc.)
            embedding: Optional new embedding vector

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Update fact in facts table
            updates["updated_at"] = "now()"
            result = (
                self.client.table("facts")
                .update(updates)
                .eq("id", fact_id)
                .eq("user_id", user_id)
                .execute()
            )

            if not result.data:
                logger.warning(f"Failed to update fact {fact_id} for user {user_id}")
                return False

            # Update embedding if provided
            if embedding:
                model_dimension = len(embedding)
                table_name = f"fact_embeddings_{model_dimension}"

                try:
                    # Try to update existing embedding
                    emb_result = (
                        self.client.table(table_name)
                        .update({"embedding": embedding})
                        .eq("fact_id", fact_id)
                        .execute()
                    )

                    # If no rows updated, insert new embedding
                    if not emb_result.data:
                        emb_result = (
                            self.client.table(table_name)
                            .insert({"fact_id": fact_id, "embedding": embedding})
                            .execute()
                        )

                except Exception as e:
                    logger.error(f"Error updating embedding for fact {fact_id}: {e}")
                    return False

            logger.info(f"Successfully updated fact {fact_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating fact {fact_id}: {e}")
            return False

    @require_auth
    def delete_fact(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete a fact and its embeddings.

        Args:
            credentials: Authentication credentials
            fact_id: Fact identifier
            user_id: User identifier for authorization

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Delete fact (embeddings will be cascade deleted due to foreign key)
            result = (
                self.client.table("facts")
                .delete()
                .eq("id", fact_id)
                .eq("user_id", user_id)
                .execute()
            )

            if not result.data:
                logger.warning(f"Failed to delete fact {fact_id} for user {user_id}")
                return False

            logger.info(f"Successfully deleted fact {fact_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting fact {fact_id}: {e}")
            return False

    @require_auth
    def check_processed_message(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: str,
        message_id: str,
    ) -> bool:
        """
        Check if a message has already been processed for fact extraction.

        Args:
            credentials: Authentication credentials
            user_id: User identifier
            message_id: Message identifier

        Returns:
            True if message has been processed, False otherwise
        """
        try:
            result = (
                self.client.table("processed_messages")
                .select("id")
                .eq("user_id", user_id)
                .eq("message_id", message_id)
                .limit(1)
                .execute()
            )

            return bool(result.data)

        except Exception as e:
            logger.error(f"Error checking processed message {message_id}: {e}")
            return False

    @require_auth
    def mark_processed_message(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: str,
        message_id: str,
        thread_id: str,
    ) -> bool:
        """
        Mark a message as processed for fact extraction.

        Args:
            credentials: Authentication credentials
            user_id: User identifier
            message_id: Message identifier
            thread_id: Thread identifier

        Returns:
            True if marked successfully, False otherwise
        """
        try:
            result = (
                self.client.table("processed_messages")
                .insert(
                    {
                        "user_id": user_id,
                        "message_id": message_id,
                        "thread_id": thread_id,
                    }
                )
                .execute()
            )

            if not result.data:
                logger.warning(
                    f"Failed to mark message {message_id} as processed for user {user_id}"
                )
                return False

            logger.debug(f"Marked message {message_id} as processed")
            return True

        except Exception as e:
            # If unique constraint violation, it's already processed - that's OK
            if "duplicate key" in str(e).lower() or "unique" in str(e).lower():
                logger.debug(f"Message {message_id} already marked as processed")
                return True
            logger.error(f"Error marking message {message_id} as processed: {e}")
            return False

    @require_auth
    def check_processed_messages_batch(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: str,
        message_ids: List[str],
    ) -> List[str]:
        """
        Check which messages have already been processed (batch mode).

        Args:
            credentials: Authentication credentials
            user_id: User identifier
            message_ids: List of message identifiers to check

        Returns:
            List of message IDs that have been processed
        """
        if not message_ids:
            return []

        try:
            result = (
                self.client.table("processed_messages")
                .select("message_id")
                .eq("user_id", user_id)
                .in_("message_id", message_ids)
                .execute()
            )

            processed_ids = [row["message_id"] for row in result.data] if result.data else []
            logger.debug(
                f"Found {len(processed_ids)} processed messages out of {len(message_ids)}"
            )
            return processed_ids

        except Exception as e:
            logger.error(f"Error checking processed messages in batch: {e}")
            return []

    @require_auth
    def mark_processed_messages_batch(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: str,
        message_data: List[Dict[str, str]],
    ) -> bool:
        """
        Mark multiple messages as processed (batch mode).

        Args:
            credentials: Authentication credentials
            user_id: User identifier
            message_data: List of dicts with 'message_id' and 'thread_id' keys

        Returns:
            True if all marked successfully, False otherwise
        """
        if not message_data:
            return True

        try:
            # Prepare batch insert data
            records = [
                {
                    "user_id": user_id,
                    "message_id": item["message_id"],
                    "thread_id": item["thread_id"],
                }
                for item in message_data
            ]

            result = self.client.table("processed_messages").insert(records).execute()

            if not result.data:
                logger.warning(
                    f"Failed to mark {len(message_data)} messages as processed for user {user_id}"
                )
                return False

            logger.debug(f"Marked {len(message_data)} messages as processed")
            return True

        except Exception as e:
            # If unique constraint violation, some messages already processed - that's OK
            if "duplicate key" in str(e).lower() or "unique" in str(e).lower():
                logger.debug(
                    "Some messages already marked as processed (duplicate key conflict)"
                )
                return True
            logger.error(f"Error marking messages as processed in batch: {e}")
            return False

    # =========================================================================
    # Fact History Methods
    # =========================================================================

    @require_auth
    def get_fact_history(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get complete history for a specific fact.

        Args:
            credentials: Authentication credentials
            fact_id: Fact identifier
            user_id: User identifier for authorization

        Returns:
            List of history records, ordered from newest to oldest
        """
        try:
            result = self.client.rpc(
                "get_fact_history",
                {"p_fact_id": fact_id, "p_user_id": user_id}
            ).execute()

            if not result.data:
                logger.debug(f"No history found for fact {fact_id}")
                return []

            logger.info(f"Found {len(result.data)} history records for fact {fact_id}")
            return result.data

        except Exception as e:
            logger.error(f"Error getting fact history for {fact_id}: {e}")
            return []

    @require_auth
    def get_recent_fact_changes(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: str,
        limit: int = 50,
        operation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent fact changes for a user.

        Args:
            credentials: Authentication credentials
            user_id: User identifier
            limit: Maximum number of records to return
            operation: Optional filter by operation type ('INSERT', 'UPDATE', 'DELETE')

        Returns:
            List of recent change records
        """
        try:
            result = self.client.rpc(
                "get_recent_fact_changes",
                {
                    "p_user_id": user_id,
                    "p_limit": limit,
                    "p_operation": operation,
                }
            ).execute()

            if not result.data:
                logger.debug(f"No recent changes found for user {user_id}")
                return []

            logger.info(f"Found {len(result.data)} recent fact changes for user {user_id}")
            return result.data

        except Exception as e:
            logger.error(f"Error getting recent fact changes for user {user_id}: {e}")
            return []

    @require_auth
    def get_fact_change_stats(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get statistics about fact changes for a user.

        Args:
            credentials: Authentication credentials
            user_id: User identifier

        Returns:
            Dictionary with change statistics:
                - total_changes: Total number of changes
                - inserts: Number of INSERT operations
                - updates: Number of UPDATE operations
                - deletes: Number of DELETE operations
                - oldest_change: Timestamp of oldest change
                - newest_change: Timestamp of newest change
        """
        try:
            result = self.client.rpc(
                "get_fact_change_stats",
                {"p_user_id": user_id}
            ).execute()

            if not result.data:
                logger.debug(f"No change stats found for user {user_id}")
                return None

            stats = result.data[0]
            logger.info(
                f"Fact change stats for user {user_id}: "
                f"{stats.get('total_changes')} total changes"
            )
            return stats

        except Exception as e:
            logger.error(f"Error getting fact change stats for user {user_id}: {e}")
            return None
