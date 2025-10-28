"""
Abstract base classes for chat storage backends.

This module defines the interface that all storage backends must implement
to ensure consistency across different database systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AnyMessage

ThreadSortBy = Literal["thread_id", "status", "created_at", "updated_at"]
SortOrder = Literal["asc", "desc"]

__all__ = ["ChatStorageBackend"]


class ChatStorageBackend(ABC):
    """Abstract base class for chat storage backends."""

    # Role mapping for database storage
    TYPE_TO_ROLE = {"human": "user", "ai": "assistant"}

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
    def ensure_thread_exists(self, thread_id: str, user_id: str) -> bool:
        """
        Ensure chat thread exists in storage.

        Args:
            thread_id: Thread identifier
            user_id: User identifier

        Returns:
            True if thread exists or was created, False otherwise
        """
        pass

    @abstractmethod
    def save_messages(
        self,
        thread_id: str,
        user_id: str,
        messages: List[AnyMessage],
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save messages to storage.

        Args:
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
