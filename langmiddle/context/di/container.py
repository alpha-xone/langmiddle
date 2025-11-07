"""Dependency injection container for automatic component wiring.

This module provides a DI container that wires components based on protocols.
No string lookups, no manual wiring, just clean dependency injection.
"""

import logging
from dataclasses import dataclass
from typing import Any

from ..config.defaults import ContextConfig
from ..core.pipeline import InjectionPipeline
from ..core.protocols import FactProcessor, FactRetriever, Formatter, StorageBackend
from ..processing.deduplication import FactDeduplicator
from ..processing.filtering import RelevanceFilter
from ..processing.formatting import ContextFormatter
from ..retrieval.global_context import GlobalFactRetriever
from ..retrieval.relevant_facts import RelevantFactRetriever
from ..retrieval.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


@dataclass
class Container:
    """DI container for automatic component wiring.

    Wires all components based on explicit dependencies.
    No string lookups, no magic, just pure dependency injection.
    """

    storage: StorageBackend
    config: ContextConfig = ContextConfig.create()
    embedder: Any | None = None  # Optional embeddings model (Embeddings)
    model: Any | None = None  # Optional LLM (BaseChatModel)

    def __post_init__(self):
        """Validate dependencies."""
        # Storage is required and must implement protocol
        assert isinstance(
            self.storage, StorageBackend
        ), f"Storage must implement StorageBackend protocol, got {type(self.storage)}"

        # Embedder and model are optional but needed for advanced features
        if self.embedder is None:
            logger.warning(
                "Embeddings model not provided. "
                "Relevant facts retrieval and summarization will fail. "
                "Provide embedder parameter to enable these features."
            )

        if self.model is None:
            logger.warning(
                "LLM model not provided. "
                "Relevance filtering and summarization will fail. "
                "Provide model parameter to enable these features."
            )

        logger.info(f"Container initialized with {type(self.storage).__name__} storage")

    # Component factories

    def global_retriever(self) -> FactRetriever:
        """Create global fact retriever."""
        return GlobalFactRetriever(storage=self.storage)

    def relevant_retriever(self) -> FactRetriever:
        """Create relevant fact retriever."""
        if not self.embedder:
            raise ValueError(
                "Embeddings model required for relevant facts retrieval. "
                "Provide embedder parameter to Container."
            )

        return RelevantFactRetriever(
            storage=self.storage,
            embedder=self.embedder,
            config=self.config.retrieval,
        )

    def deduplicator(self) -> FactProcessor:
        """Create fact deduplicator."""
        return FactDeduplicator()

    def relevance_filter(self) -> FactProcessor:
        """Create relevance filter."""
        if not self.model:
            raise ValueError(
                "LLM model required for relevance filtering. "
                "Provide model parameter to Container."
            )

        return RelevanceFilter(
            model=self.model,
            config=self.config.processing,
        )

    def formatter(self) -> Formatter:
        """Create context formatter."""
        return ContextFormatter()

    def summarizer(self) -> ConversationSummarizer:
        """Create conversation summarizer."""
        if not self.model:
            raise ValueError(
                "LLM model required for summarization. "
                "Provide model parameter to Container."
            )

        return ConversationSummarizer(
            model=self.model,
            config=self.config.summarization,
        )

    def injection_pipeline(self) -> InjectionPipeline:
        """Create injection pipeline with all dependencies wired.

        Returns:
            Fully wired injection pipeline

        Raises:
            ValueError: If required dependencies are missing
        """
        return InjectionPipeline(
            global_retriever=self.global_retriever(),
            relevant_retriever=self.relevant_retriever(),
            deduplicator=self.deduplicator(),
            relevance_filter=self.relevance_filter(),
            formatter=self.formatter(),
            summarizer=self.summarizer(),
        )


def create_container(
    storage: StorageBackend,
    embedder: Any | None = None,
    model: Any | None = None,
    config: ContextConfig | None = None,
) -> Container:
    """Factory function for creating DI container.

    Args:
        storage: Storage backend (must implement StorageBackend protocol)
        embedder: Optional embeddings model (Embeddings)
        model: Optional LLM model (BaseChatModel)
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Configured DI container

    Example:
        >>> from langmiddle.storage import SupabaseBackend
        >>> from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        >>>
        >>> storage = SupabaseBackend(url="...", key="...")
        >>> embedder = OpenAIEmbeddings()
        >>> model = ChatOpenAI()
        >>>
        >>> container = create_container(storage, embedder, model)
        >>> pipeline = container.injection_pipeline()
    """
    if config is None:
        config = ContextConfig.create()

    return Container(storage=storage, config=config, embedder=embedder, model=model)
