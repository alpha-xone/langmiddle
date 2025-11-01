"""
Firebase Firestore storage backend implementation.

This module provides Firebase Firestore-based implementation of the chat storage interface.
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger
from .base import ChatStorageBackend, SortOrder, ThreadSortBy

logger = get_graph_logger(__name__)

# Try to import Firebase dependencies
try:
    import firebase_admin
    from firebase_admin import auth, firestore
    from google.cloud.firestore import SERVER_TIMESTAMP
    from google.cloud.firestore_v1.base_query import FieldFilter

    FIREBASE_AVAILABLE = True
except ImportError as e:
    # Fail fast: require firebase-admin and google-cloud-firestore to use this backend
    raise ImportError(
        "Firebase dependencies not installed. To use the Firebase backend please install:\n"
        "  pip install firebase-admin google-cloud-firestore\n"
        "Optionally, you may also need google-auth if not already present.\n"
        f"Original error: {e}"
    )

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

    def authenticate(self, credentials: Optional[Dict[str, Any]]) -> bool:
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

        id_token = credentials.get("id_token") if credentials else None
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

    def extract_user_id(self, credentials: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Extract user ID from Firebase ID token or direct user_id.

        Args:
            credentials: Dict containing 'id_token' and/or 'user_id'

        Returns:
            User ID if found, None otherwise
        """
        if not FIREBASE_AVAILABLE:
            # Fallback to direct user_id when Firebase not available
            return credentials.get("user_id") if credentials else None

        # Check for direct user_id first
        user_id = credentials.get("user_id") if credentials else None
        if user_id:
            return user_id

        # Extract from ID token
        id_token = credentials.get("id_token") if credentials else None
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
        self,
        thread_id: str,
        user_id: str,
        messages: List[AnyMessage],
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save messages to Firestore using batch operations.

        Args:
            thread_id: Thread identifier
            user_id: User identifier
            messages: List of messages to save
            custom_state: Optional custom state defined in the graph

        Returns:
            Dict with 'saved_count' and 'errors' keys
        """
        if not FIREBASE_AVAILABLE:
            return {"saved_count": 0, "errors": ["Firebase not available"]}

        saved_count = 0
        errors = []

        if not self.ensure_thread_exists(thread_id, user_id):
            errors.append(f"Failed to ensure thread {thread_id} exists")
            return {"saved_count": saved_count, "errors": errors}

        if custom_state:
            try:
                self.db.collection("chat_threads").document(thread_id).update(
                    {"custom_state": custom_state}
                )
            except Exception as e:
                errors.append(f"Failed to update custom_state for thread {thread_id}: {e}")
                return {"saved_count": saved_count, "errors": errors}

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

    def get_thread(
        self,
        thread_id: str,
    ) -> dict | None:
        """
        Get a thread by ID.

        Args:
            thread_id: The ID of the thread to get.
        """
        if not FIREBASE_AVAILABLE:
            logger.error("Firebase not available")
            return None

        try:
            thread_ref = self.db.collection("chat_threads").document(thread_id)
            thread_doc = thread_ref.get()

            if not thread_doc.exists:
                return None

            thread_data = thread_doc.to_dict()
            return {
                "thread_id": thread_doc.id,
                "user_id": thread_data.get("user_id"),
                "created_at": thread_data.get("created_at"),
                "updated_at": thread_data.get("updated_at"),
                "custom_state": thread_data.get("custom_state"),
            }

        except Exception as e:
            logger.error(f"Error retrieving thread by ID {thread_id}: {e}")
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

        Returns:
            list[dict]: List of the threads matching the search parameters.
        """
        if not FIREBASE_AVAILABLE:
            logger.error("Firebase not available")
            return []

        try:
            query = self.db.collection("chat_threads")

            # Filter by IDs if provided
            if ids:
                query = query.where("id", "in", ids[:10])  # Firestore 'in' limit is 10
            # Apply sorting
            if sort_by:
                direction = firestore.Query.DESCENDING if sort_order == "desc" else firestore.Query.ASCENDING   # type: ignore
                query = query.order_by(sort_by, direction=direction)

            # Apply limit and offset
            query = query.limit(limit)
            if offset > 0:
                # Firestore doesn't support offset directly, this is a simplified approach
                # In production, you'd want to use cursors or document snapshots
                logger.warning("Offset not fully supported in Firebase implementation")

            docs = query.stream()

            threads = []
            for doc in docs:
                thread_data = doc.to_dict()
                thread_info = {
                    "thread_id": doc.id,
                    "user_id": thread_data.get("user_id"),
                    "created_at": thread_data.get("created_at"),
                    "updated_at": thread_data.get("updated_at"),
                    "custom_state": thread_data.get("custom_state"),
                }
                threads.append(thread_info)

            logger.debug(f"Found {len(threads)} threads matching search criteria")
            return threads

        except Exception as e:
            logger.error(f"Error searching threads: {e}")
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
        if not FIREBASE_AVAILABLE:
            logger.error("Firebase not available")
            return

        # First fetch messages for the thread
        try:
            messages_ref = self.db.collection("chat_messages")
            filter_obj = FieldFilter("thread_id", "==", thread_id)  # type: ignore
            messages_query = messages_ref.where(filter=filter_obj)
            messages_docs = list(messages_query.stream())
        except Exception as e:
            logger.error(f"Error fetching messages for deletion for thread {thread_id}: {e}")
            return

        # Now perform deletes in a separate operation / try block
        try:
            batch = self.db.batch()
            for doc in messages_docs:
                batch.delete(doc.reference)

            # Delete the thread document
            thread_ref = self.db.collection("chat_threads").document(thread_id)
            batch.delete(thread_ref)

            batch.commit()
            logger.info(f"Deleted thread {thread_id} and all its messages")
        except Exception as e:
            logger.error(f"Error committing deletion for thread {thread_id}: {e}")

    # =========================================================================
    # Facts Management Methods - Not supported in Firebase backend
    # =========================================================================

    def get_or_create_embedding_table(self, dimension: int) -> bool:
        """Ensure an embedding table exists for the given dimension."""
        raise NotImplementedError(
            "Facts management not supported in Firebase backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def insert_facts(
        self,
        user_id: str,
        facts: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
        model_dimension: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Insert facts with optional embeddings into storage."""
        raise NotImplementedError(
            "Facts management not supported in Firebase backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def query_facts(
        self,
        query_embedding: List[float],
        user_id: str,
        model_dimension: int,
        match_threshold: float = 0.75,
        match_count: int = 10,
        filter_namespaces: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Query facts using vector similarity search."""
        raise NotImplementedError(
            "Facts management not supported in Firebase backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def get_fact_by_id(
        self,
        fact_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a fact by its ID."""
        raise NotImplementedError(
            "Facts management not supported in Firebase backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def update_fact(
        self,
        fact_id: str,
        user_id: str,
        updates: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update a fact's content and/or metadata."""
        raise NotImplementedError(
            "Facts management not supported in Firebase backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )

    def delete_fact(
        self,
        fact_id: str,
        user_id: str,
    ) -> bool:
        """Delete a fact and its embeddings."""
        raise NotImplementedError(
            "Facts management not supported in Firebase backend. "
            "Use SupabaseStorageBackend for vector similarity search and facts support."
        )
