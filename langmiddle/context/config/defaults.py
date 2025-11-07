"""Default configurations for context engineering.

This module provides immutable configuration objects with sensible defaults.
Most users won't need to override anything.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for fact retrieval (immutable).

    Attributes:
        match_threshold: Minimum similarity for semantic search (0-1)
        match_count: Maximum number of facts to retrieve
        context_window: Number of recent messages to consider for context
    """

    match_threshold: float = 0.70
    match_count: int = 15
    context_window: int = 5

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.match_threshold <= 1:
            raise ValueError(f"match_threshold must be 0-1, got {self.match_threshold}")
        if self.match_count < 1:
            raise ValueError(f"match_count must be >= 1, got {self.match_count}")
        if self.context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {self.context_window}")


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration for fact processing (immutable).

    Attributes:
        enable_deduplication: Whether to remove duplicate facts
        enable_filtering: Whether to filter facts by relevance
        filter_threshold: Only filter if more than this many facts
    """

    enable_deduplication: bool = True
    enable_filtering: bool = True
    filter_threshold: int = 3

    def __post_init__(self):
        """Validate configuration."""
        if self.filter_threshold < 0:
            raise ValueError(f"filter_threshold must be >= 0, got {self.filter_threshold}")


@dataclass(frozen=True)
class SummarizationConfig:
    """Configuration for conversation summarization (immutable).

    Attributes:
        summary_threshold: Generate summary if more than this many messages
        min_messages: Minimum messages needed to generate summary
    """

    summary_threshold: int = 10
    min_messages: int = 3

    def __post_init__(self):
        """Validate configuration."""
        if self.summary_threshold < 1:
            raise ValueError(f"summary_threshold must be >= 1, got {self.summary_threshold}")
        if self.min_messages < 1:
            raise ValueError(f"min_messages must be >= 1, got {self.min_messages}")


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for memory extraction (immutable).

    Attributes:
        max_tokens_before_extraction: Extract when messages exceed this token count
                                     (None = extract on every turn)
        enable_extraction: Whether to enable memory extraction
    """

    max_tokens_before_extraction: int | None = None
    enable_extraction: bool = True


@dataclass(frozen=True)
class ContextConfig:
    """Main configuration for context engineering (immutable).

    This is the primary configuration object with sensible defaults.
    Most users only need to override 1-2 options.

    Attributes:
        max_recent_messages: Maximum number of recent messages to keep
        retrieval: Retrieval configuration
        processing: Processing configuration
        summarization: Summarization configuration
        extraction: Extraction configuration
    """

    max_recent_messages: int = 10
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    def __post_init__(self):
        """Validate configuration."""
        if self.max_recent_messages < 1:
            raise ValueError(f"max_recent_messages must be >= 1, got {self.max_recent_messages}")

    @staticmethod
    def create(
        *,
        # Queue settings
        max_recent_messages: int = 10,
        # Retrieval settings
        match_threshold: float | None = None,
        match_count: int | None = None,
        context_window: int | None = None,
        # Processing settings
        enable_deduplication: bool = True,
        enable_filtering: bool = True,
        filter_threshold: int | None = None,
        # Summarization settings
        summary_threshold: int | None = None,
        min_messages: int | None = None,
        # Extraction settings
        max_tokens_before_extraction: int | None = None,
        enable_extraction: bool = True,
    ) -> "ContextConfig":
        """Create configuration with overrides.

        This factory method provides a clean way to create config
        with only the options you want to override.

        Args:
            max_recent_messages: Max recent messages to keep
            match_threshold: Similarity threshold for retrieval
            match_count: Max facts to retrieve
            context_window: Recent messages for context
            enable_deduplication: Enable deduplication
            enable_filtering: Enable relevance filtering
            filter_threshold: Filter threshold
            summary_threshold: Summarization threshold
            min_messages: Min messages for summary
            max_tokens_before_extraction: Token threshold for extraction
            enable_extraction: Enable memory extraction

        Returns:
            Configured ContextConfig instance

        Example:
            >>> # Most users only need this
            >>> config = ContextConfig.create()

            >>> # Or override a few options
            >>> config = ContextConfig.create(
            ...     max_recent_messages=5,
            ...     match_threshold=0.75,
            ... )
        """
        # Build retrieval config
        retrieval = RetrievalConfig(
            match_threshold=match_threshold if match_threshold is not None else 0.70,
            match_count=match_count if match_count is not None else 15,
            context_window=context_window if context_window is not None else 5,
        )

        # Build processing config
        processing = ProcessingConfig(
            enable_deduplication=enable_deduplication,
            enable_filtering=enable_filtering,
            filter_threshold=filter_threshold if filter_threshold is not None else 3,
        )

        # Build summarization config
        summarization = SummarizationConfig(
            summary_threshold=summary_threshold if summary_threshold is not None else 10,
            min_messages=min_messages if min_messages is not None else 3,
        )

        # Build extraction config
        extraction = ExtractionConfig(
            max_tokens_before_extraction=max_tokens_before_extraction,
            enable_extraction=enable_extraction,
        )

        return ContextConfig(
            max_recent_messages=max_recent_messages,
            retrieval=retrieval,
            processing=processing,
            summarization=summarization,
            extraction=extraction,
        )


# Singleton default config
DEFAULT_CONFIG = ContextConfig()
