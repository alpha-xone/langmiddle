"""
Supabase storage backend implementation.

This module provides Supabase-specific implementation of the chat storage interface.
"""

import functools
import hashlib
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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


def extract_user_id_from_credentials(credentials: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Extract user ID from credentials (static utility).

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


def get_token_hash(jwt_token: Optional[str]) -> Optional[str]:
    """
    Get a hash of JWT token for comparison (avoid storing full token).

    Args:
        jwt_token: JWT token string

    Returns:
        SHA256 hash of token, or None if no token
    """
    if not jwt_token:
        return None
    return hashlib.sha256(jwt_token.encode()).hexdigest()


# =============================================================================
# DECORATORS
# =============================================================================


def with_auth_retry(func: Callable) -> Callable:
    """
    Decorator that handles authentication with smart retry logic.

    This decorator:
    1. Checks if token has changed (avoids redundant auth)
    2. Tries operation first
    3. If auth error, re-authenticates and retries once
    4. Returns appropriate error if final failure

    Args:
        func: Method to wrap with auth retry

    Returns:
        Wrapped method with auth retry logic
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract credentials from kwargs
        credentials = kwargs.get('credentials')

        # Authenticate if needed (cached internally)
        self._ensure_authenticated(credentials)

        # Try operation
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's an auth-related error
            is_auth_error = any(
                keyword in error_msg
                for keyword in ['unauthorized', 'forbidden', 'jwt', 'authentication', 'permission']
            )

            if is_auth_error:
                logger.debug(f"Auth error detected in {func.__name__}, retrying with fresh auth")

                # Force re-authentication
                self._force_authenticate(credentials)

                # Retry once
                try:
                    return func(self, *args, **kwargs)
                except Exception as retry_error:
                    logger.error(f"Auth retry failed for {func.__name__}: {retry_error}")
                    # Return appropriate error based on return type
                    return self._get_error_response(func)
            else:
                # Not an auth error, re-raise
                raise

    return wrapper


# =============================================================================
# MAIN CLASS
# =============================================================================


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
        enable_day_dreaming: bool = False,
        load_from_env: bool = True,
        user_id: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Supabase storage backend.

        Args:
            client: Existing Supabase client (optional)
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/service key
            connection_string: Direct PostgreSQL connection string (required for auto_create_tables)
            auto_create_tables: Whether to automatically create tables on initialization
            enable_facts: Whether to create facts tables (semantic memory)
            enable_day_dreaming: Whether to create fact deduplication and maintenance functions
                                 (requires enable_facts=True, includes pg_cron job setup)
            load_from_env: Whether to load credentials from environment variables
            user_id: User ID (deprecated, use credentials instead)
            credentials: Authentication credentials dict
        """
        if client:
            self.client = client
            self._current_token_hash = None
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

        # Track current authentication state (for caching)
        self._current_token_hash = None

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
                enable_day_dreaming=enable_day_dreaming,
            )

    # =========================================================================
    # AUTHENTICATION METHODS (Centralized)
    # =========================================================================

    def prepare_credentials(
        self,
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare Supabase-specific credentials."""
        credentials = {"user_id": user_id}
        if auth_token:
            credentials["jwt_token"] = auth_token
            if not user_id:
                credentials["user_id"] = self.extract_user_id(credentials)
        return credentials

    def extract_user_id(self, credentials: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract user ID from credentials."""
        return extract_user_id_from_credentials(credentials)

    def _ensure_authenticated(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """
        Ensure authentication is current (with caching).

        Only re-authenticates if token has changed.

        Args:
            credentials: Dict containing 'jwt_token' key

        Returns:
            True if authenticated or no auth needed
        """
        jwt_token = credentials.get("jwt_token") if credentials else None

        if not jwt_token:
            logger.debug("No JWT token provided, allowing non-RLS access")
            return True

        # Check if token has changed
        token_hash = get_token_hash(jwt_token)
        if token_hash == self._current_token_hash:
            logger.debug("Using cached authentication")
            return True

        # Token changed, authenticate
        return self._force_authenticate(credentials)

    def _force_authenticate(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """
        Force authentication (bypass cache).

        Args:
            credentials: Dict containing 'jwt_token' key

        Returns:
            True if authentication successful
        """
        jwt_token = credentials.get("jwt_token") if credentials else None

        if not jwt_token:
            self._current_token_hash = None
            return True

        try:
            # Set JWT on client
            self.client.postgrest.auth(jwt_token)
            self._current_token_hash = get_token_hash(jwt_token)
            logger.debug("JWT token authenticated and cached")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate JWT token: {e}")
            self._current_token_hash = None
            return False

    def authenticate(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """Public authentication method."""
        return self._ensure_authenticated(credentials)

    def invalidate_session(self) -> None:
        """Invalidate cached authentication."""
        self._current_token_hash = None
        logger.debug("Authentication cache cleared")

    def _get_error_response(self, func: Callable) -> Any:
        """Get appropriate error response based on function return type."""
        return_annotation = func.__annotations__.get('return', None)

        if return_annotation == bool:
            return False
        elif return_annotation == dict or 'Dict' in str(return_annotation):
            return {"success": False, "error": "Authentication failed"}
        elif return_annotation == list or 'List' in str(return_annotation):
            return []
        else:
            return None

    # =========================================================================
    # INTERNAL HELPER METHODS (No auth needed - used by public methods)
    # =========================================================================

    def get_existing_message_ids(self, thread_id: str) -> set:
        """Get existing message IDs from Supabase (internal helper)."""
        try:
            result = (
                self.client.table("chat_messages")
                .select("id")
                .eq("thread_id", thread_id)
                .execute()
            )

            if result.data:
                message_ids = {
                    str(msg["id"])
                    for msg in result.data
                    if isinstance(msg, dict) and "id" in msg
                }
                logger.debug(f"Found {len(message_ids)} existing messages for thread {thread_id}")
                return message_ids
            return set()
        except Exception as e:
            logger.error(f"Error getting message IDs for thread {thread_id}: {e}")
            return set()

    @with_auth_retry
    def ensure_thread_exists(self, credentials: Dict[str, Any] | None, thread_id: str, user_id: str) -> bool:
        """Ensure chat thread exists (with auth)."""
        try:
            # Ensure authenticated
            self._ensure_authenticated(credentials)

            result = (
                self.client.table("chat_threads")
                .upsert({"id": thread_id, "user_id": user_id}, on_conflict="id")
                .execute()
            )

            if not result.data:
                logger.warning(f"Thread upsert returned no data for thread {thread_id}")
                return False

            logger.debug(f"Thread {thread_id} ensured in database")
            return True
        except Exception as e:
            logger.error(f"Error ensuring thread exists: {e}")
            return False

    def get_or_create_embedding_table(self, dimension: int) -> bool:
        """Ensure embedding table exists (internal helper).

        Note: This method should only be called from within @with_auth_retry
        decorated methods to ensure proper authentication.

        Args:
            dimension: The embedding vector dimension

        Returns:
            True if table exists or was created successfully, False otherwise
        """
        try:
            # Validate dimension
            if not isinstance(dimension, int) or dimension <= 0:
                logger.error(f"Invalid dimension: {dimension}. Must be a positive integer.")
                return False

            # Call the database function to ensure table exists
            self.client.rpc("ensure_embedding_table", {"p_dimension": dimension}).execute()
            logger.debug(f"Embedding table for dimension {dimension} is ready")
            return True
        except Exception as e:
            error_msg = str(e)

            # Provide more detailed error messages
            if "function" in error_msg.lower() and "does not exist" in error_msg.lower():
                logger.error(
                    "Database function 'ensure_embedding_table' not found. "
                    "Please run the setup SQL scripts to create required functions."
                )
            elif "permission" in error_msg.lower() or "security definer" in error_msg.lower():
                logger.error(
                    f"Permission denied when creating embedding table for dimension {dimension}. "
                    f"Check database user permissions."
                )
            elif "vector" in error_msg.lower():
                logger.error(
                    f"Vector extension error for dimension {dimension}. "
                    f"Ensure pgvector extension is installed and enabled."
                )
            else:
                logger.error(f"Error creating embedding table for dimension {dimension}: {e}")

            return False

    # =========================================================================
    # CHAT OPERATIONS (With auth)
    # =========================================================================

    @with_auth_retry
    def save_messages(
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
        messages: List[AnyMessage],
        user_id: Optional[str] = None,
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save messages to Supabase."""
        user_id = user_id or self.extract_user_id(credentials)
        if not user_id:
            return {"saved_count": 0, "errors": ["user_id required"]}

        saved_count = 0
        errors = []

        if not self.ensure_thread_exists(credentials, thread_id, user_id):
            errors.append(f"Failed to ensure thread {thread_id} exists")
            return {"saved_count": saved_count, "errors": errors}

        # Update custom_state if provided
        if custom_state:
            try:
                self.client.table("chat_threads").update(
                    {"custom_state": custom_state}
                ).eq("id", thread_id).eq("user_id", user_id).execute()
            except Exception as e:
                errors.append(f"Failed to update custom_state: {e}")

        # Save messages
        for msg in messages:
            try:
                msg_data = {
                    "id": msg.id,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "content": msg.content,
                    "role": self.TYPE_TO_ROLE.get(msg.type, msg.type),
                    "metadata": getattr(msg, "response_metadata", {}),
                    "usage_metadata": getattr(msg, "usage_metadata", {}),
                }

                result = (
                    self.client.table("chat_messages")
                    .upsert(msg_data, on_conflict="id")
                    .execute()
                )

                time.sleep(0.01)  # Small delay for timestamp differentiation

                if result.data:
                    saved_count += 1
                    logger.debug(f"Saved message {msg.id}")
                else:
                    errors.append(f"Failed to save message {msg.id}")
            except Exception as e:
                errors.append(f"Error saving message {msg.id}: {e}")

        return {"saved_count": saved_count, "errors": errors}

    @with_auth_retry
    def get_thread(
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Get a thread by ID."""
        user_id = user_id or self.extract_user_id(credentials)
        if not user_id:
            logger.error("user_id required for get_thread")
            return None

        try:
            # Fetch thread
            thread = (
                self.client.table("chat_threads")
                .select("*")
                .eq("id", thread_id)
                .eq("user_id", user_id)
                .execute()
            )

            if not thread.data:
                return None

            # Fetch messages
            messages = (
                self.client.table("chat_messages")
                .select("*")
                .eq("thread_id", thread_id)
                .eq("user_id", user_id)
                .order("created_at", desc=False)
                .execute()
            )

            msgs: List[dict] = messages.data if messages.data else []  # type: ignore
            thread_dict: dict = thread.data[0]  # type: ignore
            return thread_to_dict(thread_dict, msgs)
        except Exception as e:
            logger.error(f"Error getting thread {thread_id}: {e}")
            return None

    @with_auth_retry
    def search_threads(
        self,
        credentials: Optional[Dict[str, Any]],
        *,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        values: Optional[dict] = None,
        ids: Optional[List[str]] = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: Optional[ThreadSortBy] = "updated_at",
        sort_order: Optional[SortOrder] = "desc",
    ) -> List[dict]:
        """Search for threads."""
        user_id = user_id or self.extract_user_id(credentials)
        if not user_id:
            logger.error("user_id required for search_threads")
            return []

        try:
            # Build query
            query = (
                self.client.table("chat_threads")
                .select("*")
                .eq("user_id", user_id)
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

            if not threads.data:
                return []

            # Fetch messages for all threads
            thread_list: List[dict] = threads.data  # type: ignore
            thread_ids = [str(thread["id"]) for thread in thread_list]
            messages = (
                self.client.table("chat_messages")
                .select("*")
                .in_("thread_id", thread_ids)
                .eq("user_id", user_id)
                .order("created_at", desc=False)
                .execute()
            )

            msgs: List[dict] = messages.data if messages.data else []  # type: ignore
            return [thread_to_dict(thread, msgs) for thread in thread_list]
        except Exception as e:
            logger.error(f"Error searching threads: {e}")
            return []

    @with_auth_retry
    def delete_thread(
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
        user_id: Optional[str] = None,
    ):
        """Delete a thread."""
        user_id = user_id or self.extract_user_id(credentials)
        if not user_id:
            logger.error("user_id required for delete_thread")
            return

        try:
            _ = (
                self.client.table("chat_threads")
                .delete()
                .eq("id", thread_id)
                .eq("user_id", user_id)
                .execute()
            )
            logger.info(f"Deleted thread {thread_id}")
        except Exception as e:
            logger.error(f"Error deleting thread: {e}")

    # =========================================================================
    # FACTS OPERATIONS (With auth)
    # =========================================================================

    @with_auth_retry
    def insert_facts(
        self,
        credentials: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        facts: Optional[Sequence[Dict[str, Any] | str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        model_dimension: Optional[int] = None,
        cue_embeddings: Optional[List[List[tuple[str, List[float]]]]] = None,
    ) -> Dict[str, Any]:
        """Insert facts with optional embeddings and cue embeddings.

        Args:
            credentials: Authentication credentials
            user_id: User identifier
            facts: List of facts to insert
            embeddings: List of embeddings for facts (one per fact)
            model_dimension: Embedding dimension
            cue_embeddings: Optional list of (cue_text, embedding) tuples per fact.
                           Structure: [[('cue1', emb1), ('cue2', emb2)], [('cue3', emb3)], ...]
                           Length must match facts length if provided.
        """
        user_id = user_id or self.extract_user_id(credentials)
        if not user_id:
            return {"inserted_count": 0, "fact_ids": [], "errors": ["user_id required"]}

        if not facts:
            return {"inserted_count": 0, "fact_ids": [], "errors": ["No facts provided"]}

        inserted_count = 0
        fact_ids = []
        errors = []

        # Normalize facts
        normalized_facts = []
        for idx, fact in enumerate(facts):
            if isinstance(fact, str):
                normalized_facts.append({"content": fact, "namespace": [], "language": "en"})
            elif isinstance(fact, dict):
                normalized_facts.append(fact)
            else:
                return {
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": [f"Fact at index {idx} must be string or dict"]
                }

        # Validate embeddings
        if embeddings:
            if len(embeddings) != len(normalized_facts):
                return {
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": ["Embeddings count must match facts count"]
                }

            # Validate that all embeddings are non-empty and have consistent dimensions
            if not all(embeddings):
                return {
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": ["All embeddings must be non-empty"]
                }

            embedding_dims = [len(emb) for emb in embeddings if emb]
            if not embedding_dims:
                return {
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": ["No valid embeddings provided"]
                }

            if len(set(embedding_dims)) > 1:
                return {
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": [f"Inconsistent embedding dimensions: {set(embedding_dims)}"]
                }

            if not model_dimension:
                model_dimension = embedding_dims[0]

            # Verify that all embeddings match the specified dimension
            if any(len(emb) != model_dimension for emb in embeddings):
                return {
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": [f"All embeddings must have dimension {model_dimension}"]
                }

            # Ensure embedding table exists (authentication is already handled by decorator)
            if not self.get_or_create_embedding_table(model_dimension):
                return {
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": [f"Failed to create embedding table for dimension {model_dimension}"]
                }

        # Insert facts
        for idx, fact in enumerate(normalized_facts):
            try:
                # Validate fact content
                if not fact.get("content"):
                    errors.append(f"Fact at index {idx} has no content, skipping")
                    continue

                fact_data = {
                    "user_id": user_id,
                    "content": fact.get("content"),
                    "namespace": fact.get("namespace", []),
                    "language": fact.get("language", "en"),
                    "intensity": fact.get("intensity"),
                    "confidence": fact.get("confidence"),
                    "model_dimension": model_dimension,
                }

                result = self.client.table("facts").insert(fact_data).execute()

                if not result.data:
                    errors.append(f"Failed to insert fact at index {idx}")
                    continue

                fact_record: dict = result.data[0]  # type: ignore
                fact_id = str(fact_record["id"])
                fact_ids.append(fact_id)
                inserted_count += 1

                # Insert embedding if provided
                if embeddings and idx < len(embeddings):
                    embedding = embeddings[idx]

                    # Validate embedding before insertion
                    if not embedding:
                        errors.append(f"Fact {fact_id}: Empty embedding, skipping vector insertion")
                        continue

                    if len(embedding) != model_dimension:
                        errors.append(
                            f"Fact {fact_id}: Embedding dimension {len(embedding)} "
                            f"doesn't match expected {model_dimension}, skipping"
                        )
                        continue

                    try:
                        table_name = f"fact_embeddings_{model_dimension}"
                        embedding_data = {
                            "fact_id": fact_id,
                            "user_id": user_id,
                            "embedding": embedding,
                            "fact_type": "fact",
                            "cue_text": None,
                        }
                        emb_result = self.client.table(table_name).insert(embedding_data).execute()

                        if not emb_result.data:
                            errors.append(f"Fact {fact_id}: Failed to insert embedding (no data returned)")
                        else:
                            logger.debug(f"Successfully inserted embedding for fact {fact_id}")

                        # Insert cue embeddings if provided
                        if cue_embeddings and idx < len(cue_embeddings) and cue_embeddings[idx]:
                            for cue_text, cue_embedding in cue_embeddings[idx]:
                                if not cue_embedding or len(cue_embedding) != model_dimension:
                                    errors.append(
                                        f"Fact {fact_id}: Invalid cue embedding for '{cue_text}', skipping"
                                    )
                                    continue

                                try:
                                    cue_data = {
                                        "fact_id": fact_id,
                                        "user_id": user_id,
                                        "embedding": cue_embedding,
                                        "fact_type": "cue",
                                        "cue_text": cue_text,
                                    }
                                    cue_result = self.client.table(table_name).insert(cue_data).execute()

                                    if not cue_result.data:
                                        errors.append(f"Fact {fact_id}: Failed to insert cue '{cue_text}'")
                                    else:
                                        logger.debug(f"Successfully inserted cue '{cue_text}' for fact {fact_id}")
                                except Exception as e:
                                    errors.append(f"Fact {fact_id}: Failed to insert cue '{cue_text}' - {e}")

                    except Exception as e:
                        error_msg = str(e)
                        # Provide more helpful error messages
                        if "does not exist" in error_msg.lower() or "relation" in error_msg.lower():
                            errors.append(
                                f"Fact {fact_id}: Embedding table 'fact_embeddings_{model_dimension}' not found. "
                                f"Table creation may have failed."
                            )
                        elif "permission" in error_msg.lower() or "rls" in error_msg.lower():
                            errors.append(
                                f"Fact {fact_id}: Permission denied for embedding table. "
                                f"Check RLS policies."
                            )
                        else:
                            errors.append(f"Fact {fact_id}: Failed to insert embedding - {e}")
            except Exception as e:
                errors.append(f"Error inserting fact at index {idx}: {e}")

        return {"inserted_count": inserted_count, "fact_ids": fact_ids, "errors": errors}

    @with_auth_retry
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
        """Query facts using vector similarity search or list all facts.

        Args:
            credentials: Authentication credentials
            query_embedding: Query vector for similarity search. If None, lists all facts.
            user_id: User identifier (optional, extracted from credentials if not provided)
            model_dimension: Embedding dimension (optional, inferred from query_embedding)
            match_threshold: Minimum similarity threshold (0-1, default: 0.75)
            match_count: Maximum number of results to return
            filter_namespaces: Optional list of namespace paths to filter by

        Returns:
            List of fact dictionaries with optional similarity scores
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for query_facts")
                return []

        # If query_embedding provided, infer dimension if not specified
        if query_embedding and not model_dimension:
            model_dimension = len(query_embedding)

        try:
            # If no query_embedding, list all facts (no similarity search)
            if query_embedding is None:
                # If namespace filter is provided, fetch all user facts and filter in Python
                # This is more reliable than PostgREST array comparison syntax
                if filter_namespaces:
                    query = (
                        self.client.table("facts")
                        .select("*")
                        .eq("user_id", user_id)
                        .or_(",".join(f"namespace.eq.{{{','.join(ns)}}}" for ns in filter_namespaces))
                        .limit(match_count)
                    )
                    result = query.execute()

                    if not result.data:
                        logger.debug(f"No facts found for user_id={user_id}")
                        return []

                    facts_list: List[Dict[str, Any]] = result.data  # type: ignore
                    logger.info(f"Listed {len(facts_list)} facts from {user_id} with namespace filtering")
                    return facts_list
                else:
                    # No namespace filter, just list all facts
                    query = (
                        self.client.table("facts")
                        .select("*")
                        .eq("user_id", user_id)
                        .limit(match_count)
                    )
                    result = query.execute()

                    if not result.data:
                        logger.debug(f"No facts found for user_id={user_id}")
                        return []

                    facts_list: List[Dict[str, Any]] = result.data  # type: ignore
                    logger.info(f"Listed {len(facts_list)} facts for user_id={user_id}")
                    return facts_list

            # Vector similarity search mode
            if not model_dimension:
                logger.error("model_dimension required when query_embedding is provided")
                return []

            params = {
                "p_embedding": query_embedding,
                "p_dimension": model_dimension,
                "p_user_id": user_id,
                "p_threshold": match_threshold,
                "p_limit": match_count,
                "p_namespaces": filter_namespaces if filter_namespaces else None,
            }

            result = self.client.rpc("search_facts", params).execute()

            if not result.data:
                logger.debug("No facts found matching query")
                return []

            facts_list: List[Dict[str, Any]] = result.data  # type: ignore
            logger.info(f"Found {len(facts_list)} facts matching query")

            # Log fact accesses for relevance tracking
            # Extract fact IDs and log the retrieval
            fact_ids = [fact["id"] for fact in facts_list if fact.get("id")]
            if fact_ids:
                try:
                    log_params = {
                        "p_fact_ids": fact_ids,
                        "p_user_id": user_id,
                        "p_context_type": "context",  # Default to context-based retrieval
                        "p_query_text": None,
                        "p_thread_id": None,
                    }
                    self.client.rpc("log_fact_access", log_params).execute()
                    logger.debug(f"Logged access for {len(fact_ids)} facts")
                except Exception as log_err:
                    logger.warning(f"Failed to log fact access: {log_err}")

            return facts_list
        except Exception as e:
            logger.error(f"Error querying facts: {e}")
            return []

    @with_auth_retry
    def get_fact_by_id(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a fact by its ID.

        Args:
            credentials: Authentication credentials
            fact_id: Fact identifier
            user_id: User identifier (optional, extracted from credentials if not provided)

        Returns:
            Fact dictionary if found, None otherwise
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for get_fact_by_id")
                return None

        try:
            result = (
                self.client.table("facts")
                .select("*")
                .eq("id", fact_id)
                .eq("user_id", user_id)
                .execute()
            )

            if not result.data:
                return None

            fact_record: Dict[str, Any] = result.data[0]  # type: ignore
            return fact_record
        except Exception as e:
            logger.error(f"Error getting fact {fact_id}: {e}")
            return None

    @with_auth_retry
    def update_fact(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
        updates: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update a fact's content and/or metadata.

        Args:
            credentials: Authentication credentials
            fact_id: Fact identifier
            user_id: User identifier (optional, extracted from credentials if not provided)
            updates: Dictionary of fields to update
            embedding: Optional new embedding vector

        Returns:
            True if update successful, False otherwise
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for update_fact")
                return False

        if not updates and not embedding:
            logger.warning("No updates or embedding provided for update_fact")
            return False

        try:
            # Update fact metadata if updates provided
            if updates:
                updates["updated_at"] = "now()"
                result = (
                    self.client.table("facts")
                    .update(updates)
                    .eq("id", fact_id)
                    .eq("user_id", user_id)
                    .execute()
                )

                if not result.data:
                    return False

            # Update embedding if provided
            if embedding:
                model_dimension = len(embedding)
                table_name = f"fact_embeddings_{model_dimension}"

                emb_result = (
                    self.client.table(table_name)
                    .update({"embedding": embedding})
                    .eq("fact_id", fact_id)
                    .execute()
                )

                if not emb_result.data:
                    self.client.table(table_name).insert(
                        {"fact_id": fact_id, "embedding": embedding}
                    ).execute()

            return True
        except Exception as e:
            logger.error(f"Error updating fact {fact_id}: {e}")
            return False

    @with_auth_retry
    def delete_fact(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete a fact and its embeddings.

        Args:
            credentials: Authentication credentials
            fact_id: Fact identifier
            user_id: User identifier (optional, extracted from credentials if not provided)

        Returns:
            True if deletion successful, False otherwise
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for delete_fact")
                return False

        try:
            result = (
                self.client.table("facts")
                .delete()
                .eq("id", fact_id)
                .eq("user_id", user_id)
                .execute()
            )

            if not result.data:
                return False

            logger.info(f"Deleted fact {fact_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting fact {fact_id}: {e}")
            return False

    # =========================================================================
    # PROCESSED MESSAGES (With auth)
    # =========================================================================

    @with_auth_retry
    def check_processed_message(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> bool:
        """Check if message has been processed.

        Args:
            credentials: Authentication credentials
            user_id: User identifier (optional, extracted from credentials if not provided)
            message_id: Message identifier

        Returns:
            True if message has been processed, False otherwise
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for check_processed_message")
                return False

        if not message_id:
            logger.error("message_id required for check_processed_message")
            return False

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

    @with_auth_retry
    def mark_processed_message(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> bool:
        """Mark message as processed.

        Args:
            credentials: Authentication credentials
            user_id: User identifier (optional, extracted from credentials if not provided)
            message_id: Message identifier
            thread_id: Thread identifier

        Returns:
            True if marking successful, False otherwise
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for mark_processed_message")
                return False

        if not message_id:
            logger.error("message_id required for mark_processed_message")
            return False

        if not thread_id:
            logger.error("thread_id required for mark_processed_message")
            return False

        try:
            result = (
                self.client.table("processed_messages")
                .insert({"user_id": user_id, "message_id": message_id, "thread_id": thread_id})
                .execute()
            )

            if not result.data:
                return False

            logger.debug(f"Marked message {message_id} as processed")
            return True
        except Exception as e:
            if "duplicate key" in str(e).lower() or "unique" in str(e).lower():
                logger.debug(f"Message {message_id} already processed")
                return True
            logger.error(f"Error marking message {message_id} as processed: {e}")
            return False

    @with_auth_retry
    def check_processed_messages_batch(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Check which messages have been processed (batch).

        Args:
            credentials: Authentication credentials
            user_id: User identifier (optional, extracted from credentials if not provided)
            message_ids: List of message identifiers to check

        Returns:
            List of message IDs that have been processed
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for check_processed_messages_batch")
                return []

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

            if result.data:
                rows: List[dict] = result.data  # type: ignore
                processed_ids = [str(row["message_id"]) for row in rows]
            else:
                processed_ids = []
            logger.debug(f"Found {len(processed_ids)} processed messages out of {len(message_ids)}")
            return processed_ids
        except Exception as e:
            logger.error(f"Error checking processed messages batch: {e}")
            return []

    @with_auth_retry
    def mark_processed_messages_batch(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_data: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """Mark multiple messages as processed (batch).

        Args:
            credentials: Authentication credentials
            user_id: User identifier (optional, extracted from credentials if not provided)
            message_data: List of dicts with 'message_id' and 'thread_id' keys

        Returns:
            True if marking successful, False otherwise
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for mark_processed_messages_batch")
                return False

        if not message_data:
            return True

        try:
            records = [
                {"user_id": user_id, "message_id": item["message_id"], "thread_id": item["thread_id"]}
                for item in message_data
            ]

            result = self.client.table("processed_messages").insert(records).execute()

            if not result.data:
                return False

            logger.debug(f"Marked {len(message_data)} messages as processed")
            return True
        except Exception as e:
            if "duplicate key" in str(e).lower() or "unique" in str(e).lower():
                logger.debug("Some messages already processed")
                return True
            logger.error(f"Error marking messages batch as processed: {e}")
            return False

    # =========================================================================
    # FACT HISTORY (With auth)
    # =========================================================================

    @with_auth_retry
    def get_fact_history(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get complete history for a fact.

        Args:
            credentials: Authentication credentials
            fact_id: Fact identifier
            user_id: User identifier (optional, extracted from credentials if not provided)

        Returns:
            List of history records for the fact
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for get_fact_history")
                return []

        try:
            result = self.client.rpc(
                "get_fact_history",
                {"p_fact_id": fact_id, "p_user_id": user_id}
            ).execute()

            if not result.data:
                return []

            history_list: List[Dict[str, Any]] = result.data  # type: ignore
            logger.info(f"Found {len(history_list)} history records for fact {fact_id}")
            return history_list
        except Exception as e:
            logger.error(f"Error getting fact history for {fact_id}: {e}")
            return []

    @with_auth_retry
    def get_recent_fact_changes(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        limit: int = 50,
        operation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent fact changes for a user.

        Args:
            credentials: Authentication credentials
            user_id: User identifier (optional, extracted from credentials if not provided)
            limit: Maximum number of changes to return
            operation: Optional filter by operation type (INSERT, UPDATE, DELETE)

        Returns:
            List of recent fact changes
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for get_recent_fact_changes")
                return []

        try:
            result = self.client.rpc(
                "get_recent_fact_changes",
                {"p_user_id": user_id, "p_limit": limit, "p_operation": operation}
            ).execute()

            if not result.data:
                return []

            changes_list: List[Dict[str, Any]] = result.data  # type: ignore
            logger.info(f"Found {len(changes_list)} recent fact changes")
            return changes_list
        except Exception as e:
            logger.error(f"Error getting recent fact changes: {e}")
            return []

    @with_auth_retry
    def get_fact_change_stats(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get statistics about fact changes.

        Args:
            credentials: Authentication credentials
            user_id: User identifier (optional, extracted from credentials if not provided)

        Returns:
            Dictionary with change statistics, None if error
        """
        # Extract user_id from credentials if not provided
        if not user_id:
            user_id = self.extract_user_id(credentials)
            if not user_id:
                logger.error("user_id required for get_fact_change_stats")
                return None

        try:
            result = self.client.rpc(
                "get_fact_change_stats",
                {"p_user_id": user_id}
            ).execute()

            if not result.data:
                return None

            stats: Dict[str, Any] = result.data[0]  # type: ignore
            return stats
        except Exception as e:
            logger.error(f"Error getting fact change stats: {e}")
            return None

    @with_auth_retry
    def update_fact_usage_feedback(
        self,
        credentials: Optional[Dict[str, Any]],
        access_log_ids: List[str],
        was_used: bool,
    ) -> int:
        """Update usage feedback for fact access log entries.

        Args:
            credentials: Authentication credentials
            access_log_ids: List of access log entry IDs
            was_used: Whether the facts were actually used in the response

        Returns:
            Number of records updated
        """
        try:
            result = self.client.rpc(
                "update_fact_usage_feedback",
                {"p_access_log_ids": access_log_ids, "p_was_used": was_used}
            ).execute()

            updated_count: int = int(result.data) if isinstance(result.data, (int, float)) else 0
            logger.info(f"Updated usage feedback for {updated_count} access log entries")
            return updated_count
        except Exception as e:
            logger.error(f"Error updating fact usage feedback: {e}")
            return 0

    @with_auth_retry
    def refresh_relevance_scores(
        self,
        credentials: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Refresh relevance scores for all facts.

        Args:
            credentials: Authentication credentials

        Returns:
            Dictionary with update statistics (updated_count, avg_score, min_score, max_score)
        """
        try:
            result = self.client.rpc("refresh_all_relevance_scores").execute()

            if not result.data:
                return None

            stats: Dict[str, Any] = result.data[0]  # type: ignore
            logger.info(
                f"Refreshed relevance scores: {stats.get('updated_count')} facts, "
                f"avg={stats.get('avg_score'):.3f}, "
                f"range=[{stats.get('min_score'):.3f}, {stats.get('max_score'):.3f}]"
            )
            return stats
        except Exception as e:
            logger.error(f"Error refreshing relevance scores: {e}")
            return None

    # =========================================================================
    # LEGACY/UNUSED METHOD
    # =========================================================================

    def _execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
    ) -> Optional[Any]:
        """Not supported - use Supabase query builder instead."""
        raise NotImplementedError(
            "Direct SQL execution not supported via Supabase client. "
            "Use Supabase query builder methods instead."
        )
