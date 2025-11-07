"""Builder pattern for ContextEngineer configuration."""

import logging
from typing import Self

from langchain.chat_models import init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from langchain_core.language_models import BaseChatModel

from ..storage import ChatStorage
from ..storage.base import ChatStorageBackend
from .models import ContextEngineerConfig
from .processors import ContextFormatter, FactDeduplicator, RelevanceFilter
from .queue_manager import QueueBuilder
from .retrievers import (
    ConversationSummarizer,
    GlobalContextRetriever,
    RelevantFactsRetriever,
)

logger = logging.getLogger(__name__)


class ContextEngineerBuilder:
    """Builder for creating ContextEngineer with fluent API.

    This builder provides a clean, type-safe way to configure
    ContextEngineer with sensible defaults.

    Example:
        >>> engineer = (
        ...     ContextEngineerBuilder()
        ...     .with_model("openai:gpt-4")
        ...     .with_embedder("openai:text-embedding-3-small")
        ...     .with_storage("supabase", supabase_url="...", supabase_key="...")
        ...     .with_config(max_recent_messages=5, match_threshold=0.75)
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize builder with default configuration."""
        self._config = ContextEngineerConfig()
        self._storage: ChatStorageBackend | None = None
        self._model: BaseChatModel | None = None
        self._embedder: Embeddings | None = None

    def with_model(self, model: str | BaseChatModel) -> Self:
        """Set the LLM model for extraction and filtering.

        Args:
            model: Model name (e.g., "openai:gpt-4") or BaseChatModel instance

        Returns:
            Self for chaining
        """
        if isinstance(model, str):
            self._model = init_chat_model(model, temperature=0.1)
            logger.info(f"Initialized model: {model}")
        else:
            self._model = model
            logger.info(f"Using provided model: {type(model).__name__}")
        return self

    def with_embedder(self, embedder: str | Embeddings) -> Self:
        """Set the embedding model for semantic search.

        Args:
            embedder: Embedder name (e.g., "openai:text-embedding-3-small") or Embeddings instance

        Returns:
            Self for chaining
        """
        if isinstance(embedder, str):
            self._embedder = init_embeddings(embedder)
            logger.info(f"Initialized embedder: {embedder}")
        else:
            self._embedder = embedder
            logger.info(f"Using provided embedder: {type(embedder).__name__}")
        return self

    def with_storage(self, backend: str, **kwargs) -> Self:
        """Set the storage backend for fact persistence.

        Args:
            backend: Backend type ("supabase", "postgres", "firebase", "sqlite")
            **kwargs: Backend-specific configuration (e.g., supabase_url, supabase_key)

        Returns:
            Self for chaining
        """
        storage = ChatStorage.create(backend, **kwargs)
        self._storage = storage.backend
        logger.info(f"Initialized storage backend: {backend}")
        return self

    def with_config(self, **kwargs) -> Self:
        """Set configuration options.

        Args:
            **kwargs: Configuration parameters (see ContextEngineerConfig)

        Returns:
            Self for chaining

        Example:
            >>> builder.with_config(
            ...     enable_context_injection=True,
            ...     max_recent_messages=5,
            ...     match_threshold=0.75,
            ...     enable_relevance_filter=True,
            ... )
        """
        for key, value in kwargs.items():
            if not hasattr(self._config, key):
                logger.warning(f"Unknown config option: {key}")
                continue
            setattr(self._config, key, value)
            logger.debug(f"Set config: {key} = {value}")
        return self

    def build(self):
        """Build the ContextEngineer.

        Returns:
            Configured ContextEngineer instance

        Raises:
            ValueError: If required components are not configured
        """
        # Validate required components
        if self._model is None:
            raise ValueError("Model must be configured with with_model()")
        if self._embedder is None:
            raise ValueError("Embedder must be configured with with_embedder()")
        if self._storage is None:
            raise ValueError("Storage must be configured with with_storage()")

        # Validate configuration
        self._config.__post_init__()

        # Import here to avoid circular dependency
        from ..context import ContextEngineer

        # Create components
        logger.info("Building ContextEngineer components...")

        # Retrievers
        global_retriever = GlobalContextRetriever(self._storage)
        relevant_retriever = RelevantFactsRetriever(
            self._storage,
            self._embedder,
            match_threshold=self._config.match_threshold,
            match_count=self._config.match_count,
            context_window=self._config.context_window,
        )
        summarizer = ConversationSummarizer(
            self._model,
            summary_threshold=self._config.summary_threshold,
            min_messages_to_summarize=self._config.min_messages_to_summarize,
        )

        # Processors
        deduplicator = FactDeduplicator()
        relevance_filter = RelevanceFilter(
            self._model,
            skip_threshold=self._config.relevance_filter_threshold,
        )
        formatter = ContextFormatter()

        # Queue manager
        queue_builder = QueueBuilder(self._config.max_recent_messages)

        logger.info("ContextEngineer components built successfully")

        # Create ContextEngineer (we'll pass components in a way that's compatible)
        # For now, we'll create a new-style instance
        return ContextEngineer._from_builder(
            config=self._config,
            model=self._model,
            embedder=self._embedder,
            storage=self._storage,
            global_retriever=global_retriever,
            relevant_retriever=relevant_retriever,
            summarizer=summarizer,
            deduplicator=deduplicator,
            relevance_filter=relevance_filter,
            formatter=formatter,
            queue_builder=queue_builder,
        )


class ContextComponents:
    """Container for context engineering components.

    This class groups all the modular components together
    for easier management and testing.
    """

    def __init__(
        self,
        global_retriever: GlobalContextRetriever,
        relevant_retriever: RelevantFactsRetriever,
        summarizer: ConversationSummarizer,
        deduplicator: FactDeduplicator,
        relevance_filter: RelevanceFilter,
        formatter: ContextFormatter,
        queue_builder: QueueBuilder,
    ):
        """Initialize component container.

        Args:
            global_retriever: Retriever for global context
            relevant_retriever: Retriever for relevant facts
            summarizer: Conversation summarizer
            deduplicator: Fact deduplicator
            relevance_filter: Relevance filter
            formatter: Context formatter
            queue_builder: Message queue builder
        """
        self.global_retriever = global_retriever
        self.relevant_retriever = relevant_retriever
        self.summarizer = summarizer
        self.deduplicator = deduplicator
        self.relevance_filter = relevance_filter
        self.formatter = formatter
        self.queue_builder = queue_builder
