"""Relevant facts retriever implementing FactRetriever protocol."""

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage

from ..config.defaults import RetrievalConfig
from ..core.protocols import FactRetriever, StorageBackend
from ..models import Fact

logger = logging.getLogger(__name__)


class RelevantFactRetriever:
    """Retrieves facts relevant to conversation using semantic search.

    Implements FactRetriever protocol for context-aware fact retrieval.
    Uses embeddings to find semantically similar facts.
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: Embeddings,
        config: RetrievalConfig | None = None,
    ):
        """Initialize relevant facts retriever.

        Args:
            storage: Storage backend implementing StorageBackend protocol
            embedder: Embedding model for semantic search
            config: Optional retrieval configuration (uses defaults if not provided)
        """
        self.storage = storage
        self.embedder = embedder
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        *,
        user_id: str,
        credentials: dict[str, Any],
        messages: list[BaseMessage],
        cache: dict[str, Any] | None = None,
    ) -> list[Fact]:
        """Retrieve facts relevant to conversation.

        Args:
            user_id: User identifier
            credentials: Authentication credentials
            messages: Conversation messages for context
            cache: Optional cache for embeddings

        Returns:
            List of relevant facts with similarity scores
        """
        # Extract recent content
        recent_content = self._extract_recent_content(messages)
        if not recent_content:
            logger.debug("No recent content to retrieve facts for")
            return []

        # Get or compute embedding
        context_text = " ".join(recent_content)
        try:
            embedding = self._get_embedding(context_text, cache or {})
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

        # Query storage
        try:
            results = self.storage.query_facts(
                credentials=credentials,
                query_embedding=embedding,
                user_id=user_id,
                model_dimension=len(embedding),
                match_threshold=self.config.match_threshold,
                match_count=self.config.match_count,
                filter_namespaces=None,
            )
            facts = [self._to_fact(r) for r in results]
            logger.debug(f"Retrieved {len(facts)} relevant facts")
            return facts
        except Exception as e:
            logger.error(f"Failed to query relevant facts: {e}")
            return []

    def _extract_recent_content(self, messages: list[BaseMessage]) -> list[str]:
        """Extract content from recent messages.

        Args:
            messages: List of conversation messages

        Returns:
            List of text content from recent messages
        """
        content = []
        for msg in messages[-self.config.context_window:]:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                content.append(msg.content)
        return content

    def _get_embedding(
        self,
        text: str,
        cache: dict[str, Any],
    ) -> list[float]:
        """Get embedding from cache or compute.

        Args:
            text: Text to embed
            cache: Embeddings cache

        Returns:
            Embedding vector
        """
        if text in cache:
            logger.debug("Using cached embedding")
            return cache[text]

        logger.debug("Computing new embedding")
        embedding = self.embedder.embed_query(text)
        cache[text] = embedding
        return embedding

    def _to_fact(self, result: dict[str, Any]) -> Fact:
        """Convert storage result to Fact domain model.

        Args:
            result: Raw result from storage backend

        Returns:
            Fact object with similarity score
        """
        return Fact(
            id=result["id"],
            content=result["content"],
            namespace=result["namespace"],
            confidence=result.get("confidence", 0.5),
            intensity=result.get("intensity", 0.5),
            similarity=result.get("similarity"),
        )


# Verify protocol implementation at module load time
_instance = RelevantFactRetriever.__new__(RelevantFactRetriever)
assert isinstance(_instance, FactRetriever), "RelevantFactRetriever must implement FactRetriever protocol"
