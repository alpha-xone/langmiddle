"""
Unified chat storage interface.

This module provides a unified interface for chat storage across different backends
including Supabase, SQLite, and Firebase.
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import ChatStorageBackend
from .firebase_backend import FirebaseStorageBackend
from .sqlite_backend import SQLiteStorageBackend
from .supabase_backend import SupabaseStorageBackend

logger = get_graph_logger(__name__)

__all__ = ["ChatStorage"]


class ChatStorage:
    """Unified interface for chat storage across different backends."""

    def __init__(self, backend: ChatStorageBackend):
        """
        Initialize chat storage with a specific backend.

        Args:
            backend: Storage backend implementation
        """
        self.backend = backend

    @classmethod
    def create(cls, backend_type: str, **kwargs) -> "ChatStorage":
        """
        Factory method to create storage with specific backend.

        Args:
            backend_type: Type of backend ('supabase', 'sqlite', 'firebase')
            **kwargs: Backend-specific initialization parameters

        Returns:
            ChatStorage instance with configured backend

        Raises:
            ValueError: If backend_type is not supported
        """
        backends = {
            "supabase": SupabaseStorageBackend,
            "sqlite": SQLiteStorageBackend,
            "firebase": FirebaseStorageBackend,
        }

        if backend_type not in backends:
            raise ValueError(
                f"Unknown backend: {backend_type}. "
                f"Supported backends: {list(backends.keys())}"
            )

        try:
            backend = backends[backend_type](**kwargs)
            logger.debug(f"Created {backend_type} storage backend")
            return cls(backend)
        except Exception as e:
            logger.error(f"Failed to create {backend_type} backend: {e}")
            raise

    def save_chat_history(
        self,
        thread_id: str,
        credentials: Dict[str, Any] | None,
        messages: List[AnyMessage],
        user_id: Optional[str] = None,
        saved_msg_ids: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Save chat history using the configured backend.

        Args:
            thread_id: Thread identifier for the conversation
            credentials: Authentication credentials (format varies by backend)
            messages: List of conversation messages to save
            user_id: Optional user identifier (extracted from credentials if not provided)
            saved_msg_ids: Optional set of already-saved message IDs

        Returns:
            Dict with status and info:
                - success: bool - Whether the operation succeeded
                - saved_count: int - Number of messages saved
                - errors: List[str] - Any error messages encountered
                - user_id: str - The user_id used
                - saved_msg_ids: set - Set of all saved message IDs
        """

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

        # Authenticate with backend
        if not self.backend.authenticate(credentials):
            return {
                "success": False,
                "saved_count": 0,
                "errors": ["Authentication failed"],
                "user_id": user_id,
                "saved_msg_ids": saved_msg_ids or set(),
            }

        # Extract user ID if not provided
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

        # Validate user_id
        if user_id in ["", "Invalid-User-ID", "Empty-User-ID"]:
            return {
                "success": False,
                "saved_count": 0,
                "errors": [f"Invalid user_id: {user_id}"],
                "user_id": user_id,
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
        if not self.backend.ensure_thread_exists(thread_id, user_id):
            return {
                "success": False,
                "saved_count": 0,
                "errors": ["Could not ensure thread exists"],
                "user_id": user_id,
                "saved_msg_ids": saved_msg_ids,
            }

        # Save messages
        result = self.backend.save_messages(thread_id, user_id, new_messages)

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
