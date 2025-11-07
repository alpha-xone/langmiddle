"""Context engineering module for LangMiddle.

This module provides two APIs:
1. NEW OPTIMAL API (recommended): Protocol-based, pure DI, stateless
2. OLD API: Legacy modular architecture (for backward compatibility)
"""

# OLD API (for backward compatibility)
from .builder import ContextEngineerBuilder

# NEW OPTIMAL API (recommended)
from .config.defaults import (
    ContextConfig,
    ExtractionConfig,
    ProcessingConfig,
    RetrievalConfig,
    SummarizationConfig,
)
from .core import ContextMiddleware, create_middleware
from .di import Container, create_container
from .models import ContextEngineerConfig, Fact, MessageTag, RetrievalContext
from .processors import ContextFormatter, FactDeduplicator, RelevanceFilter
from .queue_manager import MessageSeparator, MessageTagger, QueueBuilder
from .retrievers import (
    ConversationSummarizer,
    GlobalContextRetriever,
    RelevantFactsRetriever,
)

__all__ = [
    # NEW OPTIMAL API (recommended)
    "ContextMiddleware",
    "create_middleware",
    "Container",
    "create_container",
    "ContextConfig",
    "RetrievalConfig",
    "ProcessingConfig",
    "SummarizationConfig",
    "ExtractionConfig",
    # OLD API (backward compatibility)
    "Fact",
    "RetrievalContext",
    "MessageTag",
    "ContextEngineerConfig",
    "GlobalContextRetriever",
    "RelevantFactsRetriever",
    "ConversationSummarizer",
    "FactDeduplicator",
    "RelevanceFilter",
    "ContextFormatter",
    "MessageTagger",
    "MessageSeparator",
    "QueueBuilder",
    "ContextEngineerBuilder",
]
