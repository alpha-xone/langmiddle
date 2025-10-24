"""
Abstract base classes for chat storage backends.

This module defines the interface that all storage backends must implement
to ensure consistency across different database systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage

__all__ = ["ChatStorageBackend"]


class ChatStorageBackend(ABC):
    """Abstract base class for chat storage backends."""

    # Role mapping for database storage
    TYPE_TO_ROLE = {"human": "user", "ai": "assistant"}

    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with the storage backend.

        Args:
            credentials: Authentication credentials (format varies by backend)

        Returns:
            True if authentication successful, False otherwise
        """
        pass

    @abstractmethod
    def extract_user_id(self, credentials: Dict[str, Any]) -> Optional[str]:
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
        self, thread_id: str, user_id: str, messages: List[AnyMessage]
    ) -> Dict[str, Any]:
        """
        Save messages to storage.

        Args:
            thread_id: Thread identifier
            user_id: User identifier
            messages: List of messages to save

        Returns:
            Dict with 'saved_count' and 'errors' keys
        """
        pass
