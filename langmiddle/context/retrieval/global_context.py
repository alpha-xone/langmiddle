"""Global context retriever implementing FactRetriever protocol."""

import logging
from typing import Any

from langchain_core.messages import BaseMessage

from ..core.protocols import FactRetriever, StorageBackend
from ..models import Fact

logger = logging.getLogger(__name__)

# Default always-loaded namespaces for global context
ALWAYS_LOADED_NAMESPACES = [
    ["user"],
    ["user", "personal_info"],
    ["user", "professional"],
    ["user", "preferences", "*"],
]


class GlobalFactRetriever:
    """Retrieves global context from ALWAYS_LOADED_NAMESPACES.

    Implements FactRetriever protocol for global user profile facts.
    These facts are always available regardless of conversation context.
    """

    def __init__(
        self,
        storage: StorageBackend,
        namespaces: list[list[str]] | None = None,
    ):
        """Initialize global context retriever.

        Args:
            storage: Storage backend implementing StorageBackend protocol
            namespaces: List of namespace patterns to always load
        """
        self.storage = storage
        self.namespaces = namespaces or ALWAYS_LOADED_NAMESPACES

    def retrieve(
        self,
        *,
        user_id: str,
        credentials: dict[str, Any],
        messages: list[BaseMessage],
        cache: dict[str, Any] | None = None,
    ) -> list[Fact]:
        """Retrieve global context facts.

        Args:
            user_id: User identifier
            credentials: Authentication credentials
            messages: Conversation messages (not used for global context)
            cache: Optional cache (not used for global context)

        Returns:
            List of global context facts
        """
        facts: list[Fact] = []

        for namespace in self.namespaces:
            try:
                results = self.storage.query_facts(
                    credentials=credentials,
                    query_embedding=None,  # No embedding for global context
                    user_id=user_id,
                    model_dimension=None,
                    match_threshold=0.0,  # Return all matching namespace
                    match_count=10,
                    filter_namespaces=[namespace],
                )
                facts.extend([self._to_fact(r) for r in results])
            except Exception as e:
                logger.warning(f"Failed to retrieve global context for namespace {namespace}: {e}")

        logger.debug(f"Retrieved {len(facts)} global context facts")
        return facts

    def _to_fact(self, result: dict[str, Any]) -> Fact:
        """Convert storage result to Fact domain model.

        Args:
            result: Raw result from storage backend

        Returns:
            Fact object
        """
        return Fact(
            id=result["id"],
            content=result["content"],
            namespace=result["namespace"],
            confidence=result.get("confidence", 0.5),
            intensity=result.get("intensity", 0.5),
        )


# Verify protocol implementation at module load time
if not isinstance(GlobalFactRetriever, type):
    raise TypeError("GlobalFactRetriever must be a class")

# Runtime protocol check (will fail fast if protocol not implemented)
_instance = GlobalFactRetriever.__new__(GlobalFactRetriever)
assert isinstance(_instance, FactRetriever), "GlobalFactRetriever must implement FactRetriever protocol"
