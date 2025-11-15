"""
Abstract base classes for chat storage backends.

This module defines the interface that all storage backends must implement
to ensure consistency across different database systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Sequence

from langchain_core.messages import AnyMessage

ThreadSortBy = Literal["thread_id", "status", "created_at", "updated_at"]
SortOrder = Literal["asc", "desc"]

__all__ = ["ChatStorageBackend"]


class ChatStorageBackend(ABC):
    """Abstract base class for chat storage backends."""

    # Role mapping for database storage
    TYPE_TO_ROLE = {"human": "user", "ai": "assistant"}

    def prepare_credentials(
        self,
        user_id: str,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare credentials dictionary for this backend.

        Subclasses should override this to define their specific credential format.
        Default implementation provides basic user_id credential.

        Args:
            user_id: User identifier
            auth_token: Optional authentication token (JWT, Firebase ID token, etc.)

        Returns:
            Dict with backend-specific credential keys
        """
        credentials = {"user_id": user_id}
        if auth_token:
            credentials["auth_token"] = auth_token
        return credentials

    @abstractmethod
    def authenticate(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """
        Authenticate with the storage backend.

        Args:
            credentials: Authentication credentials (format varies by backend)

        Returns:
            True if authentication successful, False otherwise
        """
        pass

    @abstractmethod
    def extract_user_id(self, credentials: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Extract user ID from credentials.

        Args:
            credentials: Authentication credentials

        Returns:
            User ID if found, None otherwise
        """
        pass

    def invalidate_session(self) -> None:
        """
        Invalidate any cached session data.

        Call this when you want to force re-authentication on the next operation.
        Default implementation is a no-op for backends without session caching.
        """
        pass

    @abstractmethod
    def get_existing_message_ids(self, thread_id: str) -> set:
        """
        Get existing message IDs for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Set of existing message IDs
        """
        pass

    @abstractmethod
    def ensure_thread_exists(self, credentials: Dict[str, Any] | None, thread_id: str, user_id: str) -> bool:
        """
        Ensure a thread exists in the database.

        Args:
            credentials: Authentication credentials (format varies by backend)
            thread_id: Thread identifier
            user_id: User identifier

        Returns:
            True if thread exists or was created, False otherwise
        """
        pass

    @abstractmethod
    def save_messages(
        self,
        credentials: Optional[Dict[str, Any]],
        thread_id: str,
        user_id: str,
        messages: List[AnyMessage],
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save messages to storage.

        Args:
            credentials: Optional authentication credentials (required for RLS-enabled backends)
            thread_id: Thread identifier
            user_id: User identifier
            messages: List of messages to save
            custom_state: Optional custom state defined in the graph

        Returns:
            Dict with 'saved_count' and 'errors' keys
        """
        raise NotImplementedError("`save_messages` not implemented")

    @abstractmethod
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
        pass

    @abstractmethod
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
            headers: Optional custom headers to include with the request.

        Returns:
            list[dict]: List of the threads matching the search parameters.
        """
        raise NotImplementedError("`search_threads` not implemented.")

    @abstractmethod
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
        raise NotImplementedError("`delete_thread` not implemented")

    # =========================================================================
    # Facts Management Methods
    # =========================================================================

    @abstractmethod
    def insert_facts(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: str,
        facts: Sequence[Dict[str, Any] | str],
        embeddings: Optional[List[List[float]]] = None,
        model_dimension: Optional[int] = None,
        cue_embeddings: Optional[List[List[tuple[str, List[float]]]]] = None,
    ) -> Dict[str, Any]:
        """
        Insert facts with optional embeddings and cue embeddings into storage.

        Args:
            credentials: Authentication credentials
            user_id: User identifier
            facts: List of facts. Each fact can be either:
                - A string (auto-converted to fact dictionary)
                - A dictionary with keys: content, namespace, language, intensity, confidence
            embeddings: Optional list of embedding vectors (must match length of facts)
            model_dimension: Dimension of the embedding vectors (required if embeddings provided)
            cue_embeddings: Optional list of (cue_text, embedding) tuples per fact.
                           Structure: [[('cue1', emb1), ('cue2', emb2)], [('cue3', emb3)], ...]
                           Length must match facts length if provided.

        Returns:
            Dict with 'inserted_count', 'fact_ids', and 'errors' keys
        """
        raise NotImplementedError("`insert_facts` not implemented")

    @abstractmethod
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
        """
        Query facts using vector similarity search or list all facts.

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            query_embedding: Query vector for similarity search. If None, lists all facts.
            user_id: User identifier for filtering (optional, extracted from credentials if not provided)
            model_dimension: Dimension of the embedding model (optional, inferred from query_embedding if not provided)
            match_threshold: Minimum similarity threshold (0-1, default: 0.75). Ignored if query_embedding is None.
            match_count: Maximum number of results to return
            filter_namespaces: Optional list of namespace paths to filter by

        Returns:
            List of fact dictionaries. Includes similarity scores if query_embedding provided.
        """
        raise NotImplementedError("`query_facts` not implemented")

    @abstractmethod
    def get_fact_by_id(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a fact by its ID.

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            fact_id: Fact identifier
            user_id: User identifier for authorization (optional, extracted from credentials if not provided)

        Returns:
            Fact dictionary if found, None otherwise
        """
        raise NotImplementedError("`get_fact_by_id` not implemented")

    @abstractmethod
    def update_fact(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
        updates: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Update a fact's content and/or metadata.

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            fact_id: Fact identifier
            user_id: User identifier for authorization (optional, extracted from credentials if not provided)
            updates: Dictionary of fields to update (content, namespace, intensity, confidence, etc.)
            embedding: Optional new embedding vector

        Returns:
            True if update successful, False otherwise
        """
        raise NotImplementedError("`update_fact` not implemented")

    @abstractmethod
    def delete_fact(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a fact and its embeddings.

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            fact_id: Fact identifier
            user_id: User identifier for authorization (optional, extracted from credentials if not provided)

        Returns:
            True if deletion successful, False otherwise
        """
        raise NotImplementedError("`delete_fact` not implemented")

    def get_fact_history(
        self,
        credentials: Optional[Dict[str, Any]],
        fact_id: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get complete history for a specific fact.

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            fact_id: Fact identifier
            user_id: User identifier for authorization (optional, extracted from credentials if not provided)

        Returns:
            List of history records, ordered from newest to oldest
        """
        raise NotImplementedError("`get_fact_history` not implemented for this backend")

    def get_recent_fact_changes(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        limit: int = 50,
        operation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent fact changes for a user.

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            user_id: User identifier (optional, extracted from credentials if not provided)
            limit: Maximum number of records to return
            operation: Optional filter by operation type ('INSERT', 'UPDATE', 'DELETE')

        Returns:
            List of recent change records
        """
        raise NotImplementedError("`get_recent_fact_changes` not implemented for this backend")

    def get_fact_change_stats(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get statistics about fact changes for a user.

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            user_id: User identifier (optional, extracted from credentials if not provided)

        Returns:
            Dictionary with change statistics or None
        """
        raise NotImplementedError("`get_fact_change_stats` not implemented for this backend")

    @abstractmethod
    def check_processed_message(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> bool:
        """
        Check if a message has already been processed for fact extraction.

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            user_id: User identifier (optional, extracted from credentials if not provided)
            message_id: Message identifier

        Returns:
            True if message has been processed, False otherwise
        """
        raise NotImplementedError("`check_processed_message` not implemented")

    @abstractmethod
    def mark_processed_message(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> bool:
        """
        Mark a message as processed for fact extraction.

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            user_id: User identifier (optional, extracted from credentials if not provided)
            message_id: Message identifier
            thread_id: Thread identifier

        Returns:
            True if marked successfully, False otherwise
        """
        raise NotImplementedError("`mark_processed_message` not implemented")

    @abstractmethod
    def check_processed_messages_batch(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Check which messages have already been processed (batch mode).

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            user_id: User identifier (optional, extracted from credentials if not provided)
            message_ids: List of message identifiers to check

        Returns:
            List of message IDs that have been processed
        """
        raise NotImplementedError("`check_processed_messages_batch` not implemented")

    @abstractmethod
    def mark_processed_messages_batch(
        self,
        credentials: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        message_data: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """
        Mark multiple messages as processed (batch mode).

        Args:
            credentials: Authentication credentials containing user_id (optional if user_id provided)
            user_id: User identifier (optional, extracted from credentials if not provided)
            message_data: List of dicts with 'message_id' and 'thread_id' keys

        Returns:
            True if all marked successfully, False otherwise
        """
        raise NotImplementedError("`mark_processed_messages_batch` not implemented")

    @abstractmethod
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
        raise NotImplementedError("`get_or_create_embedding_table` not implemented")
