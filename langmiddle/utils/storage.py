"""
Database utility functions for interacting with various storage backends.

This module provides standalone utility functions for common database operations
such as saving chat history, querying threads, and managing user data.

Backward compatibility wrapper for the new storage backend system.
"""

from typing import Any, Dict, List, Optional

from jose import JWTError, jwt
from langchain_core.messages import AnyMessage

from ..storage import ChatStorage
from ..storage import save_chat_history as new_save_chat_history
from ..utils.logging import get_graph_logger

logger = get_graph_logger(__name__)

__all__ = [
    "save_chat_history",
    "extract_user_id_from_jwt",
    "ChatStorage",  # Expose new storage system
]


def extract_user_id_from_jwt(jwt_token: str) -> Optional[str]:
    """
    Extract user_id from JWT token.

    Args:
        jwt_token: JWT authentication token

    Returns:
        User ID from token payload, or None if extraction fails
    """
    if not jwt_token:
        return None

    try:
        # Decode without verification to get payload
        payload = jwt.get_unverified_claims(jwt_token)
        user_id = payload.get("sub", None)
        return user_id
    except JWTError as e:
        logger.error(f"Error decoding JWT token: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting user_id from JWT: {e}")
        return None


def save_chat_history(
    thread_id: str,
    auth_token: str | None,
    messages: List[AnyMessage],
    user_id: Optional[str] = None,
    saved_msg_ids: Optional[set] = None,
    backend_type: str = "supabase",
    **backend_kwargs,
) -> Dict[str, Any]:
    """
    Save chat history to storage backend.

    This is a backward-compatible utility function that uses the new storage
    backend system. It delegates to the appropriate backend (Supabase by default)
    while maintaining the original API.

    Args:
        thread_id: Thread identifier for the conversation
        auth_token: Optional authentication token (JWT for Supabase, ID token for Firebase, None for SQLite)
        messages: List of conversation messages to save
        user_id: Optional user identifier (extracted from auth_token if not provided, required if auth_token is None)
        saved_msg_ids: Optional set of already-saved message IDs to avoid re-querying database
        backend_type: Storage backend to use ('supabase', 'sqlite', 'firebase')
        **backend_kwargs: Additional backend-specific parameters

    Returns:
        Dict with status and info:
            - success: bool - Whether the operation succeeded
            - saved_count: int - Number of messages saved
            - errors: List[str] - Any error messages encountered
            - user_id: str - The user_id used (extracted or provided)
            - saved_msg_ids: set - Set of all saved message IDs (for persistence)

    Example:
        ```python
        from langmiddle.utils.storage import save_chat_history

        # With JWT (RLS enabled) - Supabase
        result = save_chat_history(
            thread_id="thread-123",
            auth_token="eyJhbGc...",
            messages=[message1, message2],
            user_id="user-456"  # Optional, extracted from auth_token if not provided
        )

        # Without auth token (SQLite local storage)
        result = save_chat_history(
            thread_id="thread-123",
            auth_token=None,
            messages=[message1, message2],
            user_id="user-456",
            backend_type="sqlite",
            db_path="./chat.db"
        )

        # Firebase
        result = save_chat_history(
            thread_id="thread-123",
            auth_token="firebase_id_token...",
            messages=[message1, message2],
            user_id="user-456",
            backend_type="firebase"
        )

        # With persistent tracking (e.g., in middleware)
        saved_ids = set()
        result = save_chat_history(
            thread_id="thread-123",
            auth_token="eyJhbGc...",
            messages=[message1, message2],
            saved_msg_ids=saved_ids  # Avoids re-querying database
        )
        saved_ids.update(result["saved_msg_ids"])  # Update persistent set

        if result["success"]:
            print(f"Saved {result['saved_count']} messages")
        else:
            print(f"Errors: {result['errors']}")
        ```
    """
    # Use the new storage system
    try:
        return new_save_chat_history(
            thread_id=thread_id,
            auth_token=auth_token,
            messages=messages,
            user_id=user_id,
            saved_msg_ids=saved_msg_ids,
            backend_type=backend_type,
            **backend_kwargs,
        )
    except Exception as e:
        logger.error(f"Error in save_chat_history: {e}")
        return {
            "success": False,
            "saved_count": 0,
            "errors": [f"Storage error: {e}"],
            "user_id": user_id,
            "saved_msg_ids": saved_msg_ids or set(),
        }
