"""
Firebase Firestore storage backend implementation.

This module provides Firebase Firestore-based implementation of the chat storage interface.
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import ChatStorageBackend

logger = get_graph_logger(__name__)

# Try to import Firebase dependencies
try:
    import firebase_admin
    from firebase_admin import auth, firestore
    from google.cloud.firestore import SERVER_TIMESTAMP
    from google.cloud.firestore_v1.base_query import FieldFilter

    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    # Create dummy objects to satisfy type checker
    firebase_admin = None
    firestore = None
    auth = None
    FieldFilter = None
    SERVER_TIMESTAMP = None

__all__ = ["FirebaseStorageBackend"]


class FirebaseStorageBackend(ChatStorageBackend):
    """Firebase Firestore implementation of chat storage backend."""

    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Firebase storage backend.

        Args:
            credentials_path: Optional path to Firebase service account credentials

        Raises:
            ImportError: If Firebase dependencies are not installed
            Exception: If Firebase initialization fails
        """
        if not FIREBASE_AVAILABLE:
            raise ImportError(
                "Firebase dependencies not installed. "
                "Install with: pip install firebase-admin"
            )

        try:
            if not firebase_admin._apps:  # type: ignore
                if credentials_path:
                    cred = firebase_admin.credentials.Certificate(credentials_path)  # type: ignore
                    firebase_admin.initialize_app(cred)  # type: ignore
                else:
                    firebase_admin.initialize_app()  # type: ignore

            self.db = firestore.client()  # type: ignore
            logger.debug("Firebase Firestore client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with Firebase using ID token.

        Args:
            credentials: Dict containing 'id_token' key

        Returns:
            True if authentication successful or not required
        """
        if not FIREBASE_AVAILABLE:
            logger.error("Firebase not available")
            return False

        id_token = credentials.get("id_token")
        if not id_token:
            logger.debug("No ID token provided, allowing access without authentication")
            return True  # Allow without authentication

        try:
            auth.verify_id_token(id_token)  # type: ignore
            logger.debug("Successfully authenticated with Firebase")
            return True
        except Exception as e:
            logger.error(f"Firebase authentication failed: {e}")
            return False

    def extract_user_id(self, credentials: Dict[str, Any]) -> Optional[str]:
        """
        Extract user ID from Firebase ID token or direct user_id.

        Args:
            credentials: Dict containing 'id_token' and/or 'user_id'

        Returns:
            User ID if found, None otherwise
        """
        if not FIREBASE_AVAILABLE:
            # Fallback to direct user_id when Firebase not available
            return credentials.get("user_id")

        # Check for direct user_id first
        user_id = credentials.get("user_id")
        if user_id:
            return user_id

        # Extract from ID token
        id_token = credentials.get("id_token")
        if not id_token:
            return None

        try:
            decoded_token = auth.verify_id_token(id_token)  # type: ignore
            return decoded_token.get("uid")
        except Exception as e:
            logger.error(f"Error decoding Firebase token: {e}")
            return None

    def get_existing_message_ids(self, thread_id: str) -> set:
        """
        Get existing message IDs from Firestore.

        Args:
            thread_id: Thread identifier

        Returns:
            Set of existing message IDs
        """
        if not FIREBASE_AVAILABLE:
            logger.error("Firebase not available")
            return set()

        try:
            messages_ref = self.db.collection("chat_messages")
            filter_obj = FieldFilter("thread_id", "==", thread_id)  # type: ignore
            query = messages_ref.where(filter=filter_obj)
            docs = query.stream()
            message_ids = {doc.id for doc in docs}
            logger.debug(
                f"Found {len(message_ids)} existing messages for thread {thread_id}"
            )
            return message_ids
        except Exception as e:
            logger.error(f"Error fetching existing messages: {e}")
            return set()

    def ensure_thread_exists(self, thread_id: str, user_id: str) -> bool:
        """
        Ensure chat thread exists in Firestore.

        Args:
            thread_id: Thread identifier
            user_id: User identifier

        Returns:
            True if thread exists or was created
        """
        if not FIREBASE_AVAILABLE:
            logger.error("Firebase not available")
            return False

        try:
            thread_ref = self.db.collection("chat_threads").document(thread_id)
            thread_ref.set(
                {"user_id": user_id, "created_at": SERVER_TIMESTAMP}, merge=True
            )
            logger.debug(f"Chat thread {thread_id} ensured in Firestore")
            return True
        except Exception as e:
            logger.error(f"Error ensuring thread exists: {e}")
            return False

    def save_messages(
        self, thread_id: str, user_id: str, messages: List[AnyMessage]
    ) -> Dict[str, Any]:
        """
        Save messages to Firestore using batch operations.

        Args:
            thread_id: Thread identifier
            user_id: User identifier
            messages: List of messages to save

        Returns:
            Dict with 'saved_count' and 'errors' keys
        """
        if not FIREBASE_AVAILABLE:
            return {"saved_count": 0, "errors": ["Firebase not available"]}

        saved_count = 0
        errors = []

        if not messages:
            return {"saved_count": 0, "errors": []}

        try:
            batch = self.db.batch()

            for msg in messages:
                try:
                    msg_ref = self.db.collection("chat_messages").document(msg.id)
                    msg_data = {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "content": msg.content,
                        "role": self.TYPE_TO_ROLE.get(msg.type, msg.type),
                        "metadata": getattr(msg, "response_metadata", {}),
                        "usage_metadata": getattr(msg, "usage_metadata", {}),
                        "created_at": SERVER_TIMESTAMP,
                    }

                    batch.set(msg_ref, msg_data, merge=True)
                    saved_count += 1

                except Exception as e:
                    errors.append(f"Error preparing message {msg.id}: {e}")
                    logger.error(f"Error preparing message {msg.id}: {e}")

            # Commit the batch
            if saved_count > 0:
                batch.commit()
                logger.debug(f"Saved {saved_count} messages to Firestore")
            else:
                saved_count = 0

        except Exception as e:
            errors.append(f"Error committing batch to Firestore: {e}")
            logger.error(f"Error committing batch to Firestore: {e}")
            saved_count = 0

        return {"saved_count": saved_count, "errors": errors}
