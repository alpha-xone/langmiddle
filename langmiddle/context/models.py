"""Domain models for context engineering."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage as AnyMessage


@dataclass
class Fact:
    """Domain model for a fact retrieved from storage."""

    id: str
    content: str
    namespace: list[str]
    confidence: float
    intensity: float
    similarity: float | None = None

    def __post_init__(self):
        """Validate fact data."""
        if not self.content:
            raise ValueError("Fact content cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"Intensity must be between 0 and 1, got {self.intensity}")
        if self.similarity is not None and not 0 <= self.similarity <= 1:
            raise ValueError(f"Similarity must be between 0 and 1, got {self.similarity}")


@dataclass
class RetrievalContext:
    """Context information for retrieval operations."""

    user_id: str
    credentials: dict[str, Any]
    messages: list[AnyMessage]
    embeddings_cache: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate retrieval context."""
        if not self.user_id:
            raise ValueError("user_id cannot be empty")
        if not self.credentials:
            raise ValueError("credentials cannot be empty")


class MessageTag(Enum):
    """Tags for different types of context messages."""

    GLOBAL_CONTEXT = "langmiddle/context"
    RELEVANT_FACTS = "langmiddle/facts"
    CONVERSATION_SUMMARY = "langmiddle/summary"


@dataclass
class ContextEngineerConfig:
    """Configuration for ContextEngineer."""

    # Context injection settings
    enable_context_injection: bool = True
    max_recent_messages: int = 10

    # Retrieval settings
    match_threshold: float = 0.70
    match_count: int = 15
    context_window: int = 5

    # Processing settings
    enable_deduplication: bool = True
    enable_relevance_filter: bool = True
    relevance_filter_threshold: int = 3

    # Summarization settings
    summary_threshold: int = 10
    min_messages_to_summarize: int = 3

    # Extraction settings (existing functionality)
    max_tokens_before_extraction: int | None = None
    enable_memory_extraction: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.max_recent_messages < 1:
            raise ValueError("max_recent_messages must be at least 1")
        if not 0 <= self.match_threshold <= 1:
            raise ValueError("match_threshold must be between 0 and 1")
        if self.match_count < 1:
            raise ValueError("match_count must be at least 1")
        if self.context_window < 1:
            raise ValueError("context_window must be at least 1")
