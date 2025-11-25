"""
Unified chat storage interface.

This module provides a unified interface for chat storage across different backends
including Supabase, PostgreSQL, SQLite, and Firebase.
"""

from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import ChatStorageBackend, SortOrder, ThreadSortBy
from .firebase_backend import FirebaseStorageBackend
from .postgres_backend import PostgreSQLStorageBackend
from .sqlite_backend import SQLiteStorageBackend
from .sqlite_tools import inspect_database, peek_embeddings, peek_table
from .supabase_backend import SupabaseStorageBackend

logger = get_graph_logger(__name__)

load_dotenv()

__all__ = ["ChatStorage", "inspect_database", "peek_table", "peek_embeddings"]


class ChatStorage:
    """Unified interface for chat storage across different backends.

    Supports automatic authentication for 'langmiddle' backend.
    """

    def __init__(self, backend: ChatStorageBackend, backend_type: str = "unknown"):
        """
        Initialize chat storage with a specific backend.

        Args:
            backend: Storage backend implementation
            backend_type: Type of backend (for internal tracking)
        """
        self.backend = backend
        self.backend_type = backend_type
        self._auth_token: str | None = None
        self._user_id: str | None = None

    @classmethod
    def create(cls, backend_type: str, **kwargs) -> "ChatStorage":
        """
        Factory method to create storage with specific backend.

        Args:
            backend_type: Type of backend ('langmiddle', 'supabase', 'postgres', 'sqlite', 'firebase')
            **kwargs: Backend-specific initialization parameters

        Returns:
            ChatStorage instance with configured backend

        Raises:
            ValueError: If backend_type is not supported
        """
        # Handle langmiddle platform backend with automatic authentication
        if backend_type == "langmiddle":
            import os

            from ..auth.session import ensure_authenticated

            credentials = ensure_authenticated()

            # Use shared backend URL and key
            kwargs.setdefault("supabase_url", os.getenv(
                "LANGMIDDLE_PROJECT_URL",
                "https://frfdkegapvchcnczyesb.supabase.co"
            ))
            kwargs.setdefault("supabase_key", os.getenv(
                "LANGMIDDLE_ANON_KEY",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZyZmRrZWdhcHZjaGNuY3p5ZXNiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMxNTE1MjcsImV4cCI6MjA3ODcyNzUyN30.8br0FE9Dbk0dmiPCz8Lqzi4rb0r9vO_P6VU5S8uOtNk"
            ))

            # Create backend
            backend = SupabaseStorageBackend(**kwargs)
            storage = cls(backend, "langmiddle")

            # Store auth info for automatic injection
            storage._auth_token = credentials["access_token"]
            storage._user_id = credentials["user_id"]

            logger.info(f"Created LangMiddle platform backend for user: {storage._user_id}")
            return storage

        backends = {
            "supabase": SupabaseStorageBackend,
            "postgres": PostgreSQLStorageBackend,
            "postgresql": PostgreSQLStorageBackend,
            "sqlite": SQLiteStorageBackend,
            "firebase": FirebaseStorageBackend,
            "langmiddle": SupabaseStorageBackend,  # Alias for langmiddle platform
        }

        if backend_type not in backends:
            raise ValueError(
                f"Unknown backend: {backend_type}. "
                f"Supported backends: {list(backends.keys())}"
            )

        try:
            backend = backends[backend_type](**kwargs)
            logger.debug(f"Created {backend_type} storage backend")
            return cls(backend, backend_type)
        except Exception as e:
            logger.error(f"Failed to create {backend_type} backend: {e}")
            raise

    def _inject_auth(self, credentials: Dict[str, Any] | None) -> Dict[str, Any]:
        """Inject authentication token for LangMiddle platform backend.

        Args:
            credentials: Original credentials dictionary

        Returns:
            Credentials with auth token injected if using langmiddle backend
        """
        if self.backend_type == "langmiddle":
            credentials = credentials.copy() if credentials else {}
            credentials["auth_token"] = self._auth_token
            if "user_id" not in credentials:
                if not self._user_id:
                    self._user_id = self.backend.extract_user_id(credentials)
                if self._user_id:
                    credentials["user_id"] = self._user_id
        return credentials or {}

    def save_chat_history(
        self,
        thread_id: str,
        messages: List[AnyMessage],
        user_id: str | None = None,
        saved_msg_ids: set | None = None,
        custom_state: Dict[str, Any] | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Save chat history using the configured backend.

        Args:
            thread_id: Thread identifier for the conversation
            messages: List of conversation messages to save
            user_id: Optional user identifier (extracted from credentials if not provided)
            saved_msg_ids: Optional set of already-saved message IDs
            custom_state: Optional custom state defined in the graph
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            Dict with status and info:
                - success: bool - Whether the operation succeeded
                - saved_count: int - Number of messages saved
                - errors: List[str] - Any error messages encountered
                - user_id: str - The user_id used
                - saved_msg_ids: set - Set of all saved message IDs
        """
        credentials = self._inject_auth(credentials)

        # Validate inputs
        if not thread_id:
            return {
                "success": False,
                "saved_count": 0,
                "errors": ["thread_id is required"],
                "user_id": None,
                "saved_msg_ids": saved_msg_ids or set(),
            }

        if not messages:
            logger.debug(f"No messages to save for thread {thread_id}")
            return {
                "success": True,
                "saved_count": 0,
                "errors": [],
                "user_id": user_id,
                "saved_msg_ids": saved_msg_ids or set(),
            }

        # Extract user_id if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            return {
                "success": False,
                "saved_count": 0,
                "errors": ["Could not determine user_id"],
                "user_id": None,
                "saved_msg_ids": saved_msg_ids or set(),
            }

        # Get existing message IDs if not provided
        if saved_msg_ids is None:
            saved_msg_ids = self.backend.get_existing_message_ids(thread_id)
        else:
            logger.debug(
                f"Using provided saved_msg_ids set with {len(saved_msg_ids)} existing messages"
            )

        # Filter out already saved messages
        new_messages = [msg for msg in messages if msg.id not in saved_msg_ids]

        if not new_messages:
            logger.debug(f"All messages already saved for thread {thread_id}")
            return {
                "success": True,
                "saved_count": 0,
                "errors": [],
                "user_id": user_id,
                "total_messages": len(messages),
                "skipped_count": len(messages),
                "saved_msg_ids": saved_msg_ids,
            }

        # Ensure thread exists
        if not self.backend.ensure_thread_exists(user_id=user_id, thread_id=thread_id, credentials=credentials):
            return {
                "success": False,
                "saved_count": 0,
                "errors": ["Could not ensure thread exists"],
                "user_id": user_id,
                "saved_msg_ids": saved_msg_ids,
            }

        # Save messages via backend (passes credentials internally if supported)
        result = self.backend.save_messages(
            thread_id=thread_id,
            messages=new_messages,
            user_id=user_id,
            custom_state=custom_state,
            credentials=credentials,
        )

        # Update saved message IDs for successfully saved messages
        successfully_saved = new_messages[: result["saved_count"]]
        for msg in successfully_saved:
            saved_msg_ids.add(msg.id)

        # Determine overall success
        success = result["saved_count"] > 0 or len(result["errors"]) == 0

        return {
            "success": success,
            "saved_count": result["saved_count"],
            "errors": result["errors"],
            "user_id": user_id,
            "total_messages": len(messages),
            "skipped_count": len(messages) - len(new_messages),
            "saved_msg_ids": saved_msg_ids,
        }

    def get_thread(
        self,
        thread_id: str,
        credentials: Dict[str, Any] | None,
    ) -> dict | None:
        """
        Get a thread.

        Args:
            thread_id: Thread identifier for the conversation
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            dict: The thread matching the thread_id, or None if not found.
        """
        # Pass credentials to backend - backends with auth will use decorator
        credentials = self._inject_auth(credentials)
        return self.backend.get_thread(
            thread_id=thread_id,
            credentials=credentials,
        )

    def search_threads(
        self,
        metadata: dict | None = None,
        values: dict | None = None,
        ids: List[str] | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: ThreadSortBy | None = "updated_at",
        sort_order: SortOrder | None = "desc",
        credentials: Dict[str, Any] | None = None,
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
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            list[dict]: List of the threads matching the search parameters.
        """
        # Inject auth and authenticate with backend
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error(f"Authentication failed with credentials: {credentials}")
            return []

        return self.backend.search_threads(
            metadata=metadata,
            values=values,
            ids=ids,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    # =========================================================================
    # Facts Management Methods
    # =========================================================================

    def get_or_create_embedding_table(
        self,
        dimension: int,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Ensure an embedding table exists for the given dimension.

        Args:
            dimension: Embedding vector dimension (e.g., 1536 for OpenAI, 768 for sentence-transformers)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            True if table exists or was created, False otherwise
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for get_or_create_embedding_table")
            return False

        return self.backend.get_or_create_embedding_table(dimension)

    def insert_facts(
        self,
        facts: Sequence[Dict[str, Any] | str],
        embeddings: List[List[float]] | None = None,
        model_dimension: int | None = None,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Insert facts with optional embeddings into storage.

        Args:
            facts: List of facts. Each fact can be either:
                - A string (auto-converted to {"content": string, "namespace": [], "language": "en"})
                - A dictionary with keys:
                  - content: str (required) - The fact content
                  - namespace: List[str] (optional) - Hierarchical namespace path
                  - language: str (optional) - Language code (default: 'en')
                  - intensity: float (optional) - Intensity score 0-1
                  - confidence: float (optional) - Confidence score 0-1
            embeddings: Optional list of embedding vectors (must match facts length)
            model_dimension: Embedding dimension (optional - will be inferred from embeddings if not provided)
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            Dict with status:
                - success: bool - Whether operation succeeded
                - inserted_count: int - Number of facts inserted
                - fact_ids: List[str] - IDs of inserted facts
                - errors: List[str] - Any error messages

        Note:
            - Facts can be passed as simple strings for convenience
            - If embeddings are provided, all vectors must have the same dimension
            - model_dimension will be inferred from the first embedding if not explicitly provided
            - Validation ensures dimensional consistency across all embeddings
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            return {
                "success": False,
                "inserted_count": 0,
                "fact_ids": [],
                "errors": ["Authentication failed"],
            }

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            return {
                "success": False,
                "inserted_count": 0,
                "fact_ids": [],
                "errors": ["Could not determine user_id"],
            }

        # Normalize facts: convert strings to dictionaries for convenience
        normalized_facts: List[Dict[str, Any]] = []
        for idx, fact in enumerate(facts):
            if isinstance(fact, str):
                # Auto-convert string to fact dictionary
                normalized_facts.append({
                    "content": fact,
                    "namespace": [],
                    "language": "en",
                })
            elif isinstance(fact, dict):
                normalized_facts.append(fact)
            else:
                return {
                    "success": False,
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": [
                        f"Fact at index {idx} must be a string or dictionary, got {type(fact).__name__}"
                    ],
                }

        # After normalization, all facts are guaranteed to be dictionaries
        facts_dicts: List[Dict[str, Any]] = normalized_facts

        # Validate embeddings if provided
        if embeddings is not None:
            if len(embeddings) != len(facts_dicts):
                return {
                    "success": False,
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": [
                        f"Embeddings count ({len(embeddings)}) must match facts count ({len(facts_dicts)})"
                    ],
                }

            # Validate embeddings are not empty
            if not embeddings:
                return {
                    "success": False,
                    "inserted_count": 0,
                    "fact_ids": [],
                    "errors": ["Embeddings list cannot be empty"],
                }

            # Infer model_dimension from first embedding if not provided
            if not model_dimension:
                model_dimension = len(embeddings[0])
                logger.debug(f"Inferred model_dimension={model_dimension} from embeddings")

            # Validate all embeddings have consistent dimensions
            for i, embedding in enumerate(embeddings):
                if not embedding:
                    return {
                        "success": False,
                        "inserted_count": 0,
                        "fact_ids": [],
                        "errors": [f"Embedding at index {i} is empty"],
                    }
                if len(embedding) != model_dimension:
                    return {
                        "success": False,
                        "inserted_count": 0,
                        "fact_ids": [],
                        "errors": [
                            f"Embedding at index {i} has dimension {len(embedding)}, "
                            f"expected {model_dimension}. All embeddings must have the same dimension."
                        ],
                    }

        return self.backend.insert_facts(
            facts=facts_dicts,
            embeddings=embeddings,
            model_dimension=model_dimension,
            user_id=user_id,
            credentials=credentials,
        )

    def query_facts(
        self,
        query_embedding: List[float],
        model_dimension: int,
        match_threshold: float = 0.75,
        match_count: int = 10,
        filter_namespaces: List[List[str]] | None = None,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Query facts using vector similarity search.

        Args:
            query_embedding: Query embedding vector
            model_dimension: Dimension of the embedding model
            match_threshold: Minimum similarity threshold (0-1, default: 0.75)
            match_count: Maximum number of results to return
            filter_namespaces: Optional list of namespace paths to filter by
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            List of fact dictionaries with similarity scores, sorted by relevance
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for query_facts")
            return []

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for query_facts")
            return []

        # Validate query embedding dimension
        if not query_embedding:
            logger.error("Query embedding is empty")
            return []

        if len(query_embedding) != model_dimension:
            logger.error(
                f"Query embedding dimension ({len(query_embedding)}) does not match "
                f"model_dimension ({model_dimension})"
            )
            return []

        return self.backend.query_facts(
            query_embedding=query_embedding,
            user_id=user_id,
            model_dimension=model_dimension,
            match_threshold=match_threshold,
            match_count=match_count,
            filter_namespaces=filter_namespaces,
            credentials=credentials,
        )

    def get_fact_by_id(
        self,
        fact_id: str,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """
        Get a fact by its ID.

        Args:
            fact_id: Fact identifier
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            Fact dictionary if found, None otherwise
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for get_fact_by_id")
            return None

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for get_fact_by_id")
            return None

        return self.backend.get_fact_by_id(
            fact_id=fact_id,
            user_id=user_id,
            credentials=credentials,
        )

    def update_fact(
        self,
        fact_id: str,
        updates: Dict[str, Any],
        embedding: List[float] | None = None,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Update a fact's content and/or metadata.

        Args:
            fact_id: Fact identifier
            updates: Dictionary of fields to update:
                - content: str - Update fact content
                - namespace: List[str] - Update namespace
                - language: str - Update language
                - intensity: float - Update intensity (0-1)
                - confidence: float - Update confidence (0-1)
            embedding: Optional new embedding vector (requires model_dimension in fact)
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            True if update successful, False otherwise
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for update_fact")
            return False

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for update_fact")
            return False

        # Validate embedding dimension if provided
        if embedding is not None:
            if not embedding:
                logger.error("Embedding cannot be empty")
                return False

            # Get the existing fact to verify dimension matches
            existing_fact = self.backend.get_fact_by_id(
                fact_id=fact_id,
                user_id=user_id,
                credentials=credentials,
            )
            if existing_fact and "model_dimension" in existing_fact:
                expected_dimension = existing_fact["model_dimension"]
                if len(embedding) != expected_dimension:
                    logger.error(
                        f"Embedding dimension ({len(embedding)}) does not match "
                        f"fact's model_dimension ({expected_dimension})"
                    )
                    return False

        return self.backend.update_fact(
            fact_id=fact_id,
            user_id=user_id,
            updates=updates,
            embedding=embedding,
            credentials=credentials,
        )

    def delete_fact(
        self,
        fact_id: str,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Delete a fact and its embeddings.

        Args:
            fact_id: Fact identifier
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            True if deletion successful, False otherwise
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for delete_fact")
            return False

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for delete_fact")
            return False

        return self.backend.delete_fact(
            fact_id=fact_id,
            user_id=user_id,
            credentials=credentials,
        )

    # =========================================================================
    # Processed Messages Tracking
    # =========================================================================

    def check_processed_message(
        self,
        message_id: str,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Check if a message has already been processed for fact extraction.

        Args:
            message_id: Message identifier
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            True if message has been processed, False otherwise
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for check_processed_message")
            return False

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for check_processed_message")
            return False

        return self.backend.check_processed_message(
            user_id=user_id,
            message_id=message_id,
            credentials=credentials,
        )

    def mark_processed_message(
        self,
        message_id: str,
        thread_id: str,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Mark a message as processed for fact extraction.

        Args:
            message_id: Message identifier
            thread_id: Thread identifier
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            True if marked successfully, False otherwise
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for mark_processed_message")
            return False

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for mark_processed_message")
            return False

        return self.backend.mark_processed_message(
            user_id=user_id,
            message_id=message_id,
            thread_id=thread_id,
            credentials=credentials,
        )

    def check_processed_messages_batch(
        self,
        message_ids: List[str],
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> List[str]:
        """
        Check which messages have already been processed (batch mode).

        Args:
            message_ids: List of message identifiers to check
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            List of message IDs that have been processed
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for check_processed_messages_batch")
            return []

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for check_processed_messages_batch")
            return []

        return self.backend.check_processed_messages_batch(
            user_id=user_id,
            message_ids=message_ids,
            credentials=credentials,
        )

    def mark_processed_messages_batch(
        self,
        message_data: List[Dict[str, str]],
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Mark multiple messages as processed (batch mode).

        Args:
            message_data: List of dicts with 'message_id' and 'thread_id' keys
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            True if all marked successfully, False otherwise
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for mark_processed_messages_batch")
            return False

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for mark_processed_messages_batch")
            return False

        return self.backend.mark_processed_messages_batch(
            user_id=user_id,
            message_data=message_data,
            credentials=credentials,
        )

    # =========================================================================
    # Fact History Methods
    # =========================================================================

    def get_fact_history(
        self,
        fact_id: str,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get complete history for a specific fact.

        Args:
            fact_id: Fact identifier
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            List of history records, ordered from newest to oldest
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for get_fact_history")
            return []

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for get_fact_history")
            return []

        # Check if backend supports fact history
        if not hasattr(self.backend, 'get_fact_history'):
            logger.warning(f"Backend {type(self.backend).__name__} does not support fact history")
            return []

        return self.backend.get_fact_history(
            fact_id=fact_id,
            user_id=user_id,
            credentials=credentials,
        )

    def get_recent_fact_changes(
        self,
        limit: int = 50,
        operation: str | None = None,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent fact changes for a user.

        Args:
            limit: Maximum number of records to return (default: 50)
            operation: Optional filter by operation type ('INSERT', 'UPDATE', 'DELETE')
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            List of recent change records
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for get_recent_fact_changes")
            return []

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for get_recent_fact_changes")
            return []

        # Check if backend supports fact history
        if not hasattr(self.backend, 'get_recent_fact_changes'):
            logger.warning(f"Backend {type(self.backend).__name__} does not support fact history")
            return []

        return self.backend.get_recent_fact_changes(
            user_id=user_id,
            limit=limit,
            operation=operation,
            credentials=credentials,
        )

    def get_fact_change_stats(
        self,
        user_id: str | None = None,
        credentials: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """
        Get statistics about fact changes for a user.

        Args:
            user_id: Optional user identifier (extracted from credentials if not provided)
            credentials: Optional authentication credentials (format varies by backend)

        Returns:
            Dictionary with change statistics:
                - total_changes: Total number of changes
                - inserts: Number of INSERT operations
                - updates: Number of UPDATE operations
                - deletes: Number of DELETE operations
                - oldest_change: Timestamp of oldest change
                - newest_change: Timestamp of newest change
        """
        credentials = self._inject_auth(credentials)
        if not self.backend.authenticate(credentials):
            logger.error("Authentication failed for get_fact_change_stats")
            return None

        # Extract user ID if not provided
        if not user_id:
            user_id = self.backend.extract_user_id(credentials)

        if not user_id:
            logger.error("Could not determine user_id for get_fact_change_stats")
            return None

        # Check if backend supports fact history
        if not hasattr(self.backend, 'get_fact_change_stats'):
            logger.warning(f"Backend {type(self.backend).__name__} does not support fact history")
            return None

        return self.backend.get_fact_change_stats(
            user_id=user_id,
            credentials=credentials,
        )

    def invalidate_session(self) -> None:
        """
        Invalidate any cached session data (JWT tokens, user_id, etc.).

        Call this when you want to force re-authentication on the next operation,
        for example:
        - When switching between different users
        - After obtaining a new JWT token (e.g., after refresh)
        - When you want to ensure fresh authentication

        Example:
            ```python
            storage = ChatStorage.create("supabase", ...)

            # Use with first user
            credentials1 = storage.backend.prepare_credentials("user1", jwt1)
            storage.save_chat_history(thread_id, credentials1, messages)

            # Switch to different user - invalidate cached session
            storage.invalidate_session()
            credentials2 = storage.backend.prepare_credentials("user2", jwt2)
            storage.save_chat_history(thread_id, credentials2, messages)
            ```
        """
        self.backend.invalidate_session()
