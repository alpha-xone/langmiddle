"""Context retrieval strategies."""

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage as AnyMessage
from langchain_core.messages import SystemMessage

from ..storage.base import ChatStorageBackend
from .models import Fact, RetrievalContext

logger = logging.getLogger(__name__)

# Default always-loaded namespaces for global context
ALWAYS_LOADED_NAMESPACES = [
    ["user", "personal_info"],
    ["user", "professional"],
    ["user", "preferences", "*"],
]


class GlobalContextRetriever:
    """Retrieves global context from ALWAYS_LOADED_NAMESPACES.

    This retriever fetches facts that should always be available,
    such as user profile information and preferences.
    """

    def __init__(
        self,
        storage_backend: ChatStorageBackend,
        namespaces: list[list[str]] | None = None,
    ):
        """Initialize global context retriever.

        Args:
            storage_backend: Storage backend for querying facts
            namespaces: List of namespace patterns to always load.
                        Defaults to ALWAYS_LOADED_NAMESPACES.
        """
        self.storage = storage_backend
        self.namespaces = namespaces or ALWAYS_LOADED_NAMESPACES

    def retrieve(self, context: RetrievalContext) -> list[Fact]:
        """Retrieve global context facts.

        Args:
            context: Retrieval context with user_id and credentials

        Returns:
            List of facts from global namespaces
        """
        facts = []

        for namespace in self.namespaces:
            try:
                results = self.storage.query_facts(
                    credentials=context.credentials,
                    query_embedding=None,
                    user_id=context.user_id,
                    model_dimension=None,
                    match_threshold=0.0,
                    match_count=10,
                    filter_namespaces=[namespace],
                )
                facts.extend([self._to_fact(r) for r in results])
            except Exception as e:
                logger.warning(
                    f"Failed to retrieve global context for namespace {namespace}: {e}"
                )

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


class RelevantFactsRetriever:
    """Retrieves facts relevant to recent conversation using semantic search.

    This retriever embeds recent conversation content and finds
    semantically similar facts from the storage backend.
    """

    def __init__(
        self,
        storage_backend: ChatStorageBackend,
        embedder: Embeddings,
        match_threshold: float = 0.70,
        match_count: int = 15,
        context_window: int = 5,
    ):
        """Initialize relevant facts retriever.

        Args:
            storage_backend: Storage backend for querying facts
            embedder: Embedding model for semantic search
            match_threshold: Minimum similarity threshold (0-1)
            match_count: Maximum number of facts to retrieve
            context_window: Number of recent messages to consider
        """
        self.storage = storage_backend
        self.embedder = embedder
        self.match_threshold = match_threshold
        self.match_count = match_count
        self.context_window = context_window

    def retrieve(self, context: RetrievalContext) -> list[Fact]:
        """Retrieve facts relevant to conversation.

        Args:
            context: Retrieval context with messages and embeddings cache

        Returns:
            List of relevant facts with similarity scores
        """
        # Extract recent content
        recent_content = self._extract_recent_content(context.messages)
        if not recent_content:
            logger.debug("No recent content to retrieve facts for")
            return []

        # Get or compute embedding
        context_text = " ".join(recent_content)
        try:
            embedding = self._get_embedding(context_text, context.embeddings_cache)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

        # Query storage
        try:
            results = self.storage.query_facts(
                credentials=context.credentials,
                query_embedding=embedding,
                user_id=context.user_id,
                model_dimension=len(embedding),
                match_threshold=self.match_threshold,
                match_count=self.match_count,
                filter_namespaces=None,
            )
            return [self._to_fact(r) for r in results]
        except Exception as e:
            logger.error(f"Failed to query relevant facts: {e}")
            return []

    def _extract_recent_content(self, messages: list[AnyMessage]) -> list[str]:
        """Extract content from recent messages.

        Args:
            messages: List of conversation messages

        Returns:
            List of text content from recent messages
        """
        content = []
        for msg in messages[-self.context_window:]:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                content.append(msg.content)
        return content

    def _get_embedding(
        self,
        text: str,
        cache: dict[str, list[float]],
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


class ConversationSummarizer:
    """Generates summaries of conversation history using LLM.

    This summarizer creates concise summaries of older conversation
    context to reduce token usage while preserving important information.
    """

    def __init__(
        self,
        model: BaseChatModel,
        summary_threshold: int = 10,
        min_messages_to_summarize: int = 3,
    ):
        """Initialize conversation summarizer.

        Args:
            model: LLM for generating summaries
            summary_threshold: Only summarize if more than this many messages
            min_messages_to_summarize: Minimum messages needed for summary
        """
        self.model = model
        self.summary_threshold = summary_threshold
        self.min_messages = min_messages_to_summarize

    def summarize(self, messages: list[AnyMessage]) -> str | None:
        """Generate conversation summary if needed.

        Args:
            messages: Full conversation history

        Returns:
            Summary text or None if not needed
        """
        # Check if summarization is needed
        if len(messages) <= self.summary_threshold:
            logger.debug(
                f"Skipping summarization: {len(messages)} messages <= {self.summary_threshold} threshold"
            )
            return None

        # Extract messages to summarize (exclude recent ones)
        messages_to_summarize = messages[:-self.summary_threshold]
        if len(messages_to_summarize) < self.min_messages:
            logger.debug(
                f"Not enough messages to summarize: {len(messages_to_summarize)} < {self.min_messages}"
            )
            return None

        # Generate summary
        try:
            prompt = self._build_prompt(messages_to_summarize)
            response = self.model.invoke([SystemMessage(content=prompt)])
            summary = self._extract_content(response)
            logger.info(f"Generated summary of {len(messages_to_summarize)} messages")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return None

    def _build_prompt(self, messages: list[AnyMessage]) -> str:
        """Build summarization prompt.

        Args:
            messages: Messages to summarize

        Returns:
            Prompt text
        """
        # Format messages for prompt
        formatted_messages = []
        for msg in messages:
            role = msg.__class__.__name__.replace("Message", "")
            content = getattr(msg, "content", str(msg))
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            formatted_messages.append(f"{role}: {content}")

        return f"""Summarize the key points from this conversation history concisely.
Focus on: decisions made, information provided, user preferences expressed, and ongoing tasks.

Conversation to summarize:
{chr(10).join(formatted_messages)}

Provide a clear, structured summary (3-5 bullet points max):"""

    def _extract_content(self, response: Any) -> str:
        """Extract text content from model response.

        Args:
            response: Model response object

        Returns:
            Text content
        """
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                return " ".join(str(item) for item in content)
            return str(content)
        return str(response)
