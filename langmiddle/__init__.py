"""Middlewares for LangChain / LangGraph

This package provides middleware components for LangChain and LangGraph applications,
enabling enhanced functionality and streamlined development workflows.
"""

__version__ = "0.1.4a1"
__author__ = "Alpha x1"
__email__ = "alpha.xone@outlook.com"
__x__ = "alpha_xone_"

# NEW OPTIMAL CONTEXT API (recommended)
from .context import (
    Container,
    ContextConfig,
    ContextMiddleware,
    create_container,
    create_middleware,
)

# Modular context engineering components (legacy/backward compatibility)
from .context.builder import ContextEngineerBuilder
from .context.models import (
    ContextEngineerConfig,
    Fact,
    MessageTag,
    RetrievalContext,
)
from .context.processors import (
    ContextFormatter,
    FactDeduplicator,
    RelevanceFilter,
)
from .context.queue_manager import (
    MessageSeparator,
    MessageTagger,
    QueueBuilder,
)
from .context.retrievers import (
    ConversationSummarizer,
    GlobalContextRetriever,
    RelevantFactsRetriever,
)

# Main middleware classes
from .history import ChatSaver, StorageContext, ToolRemover

# Storage backends
from .storage import ChatStorage

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__x__",
    # Main classes
    "ChatSaver",
    "StorageContext",
    "ToolRemover",
    "ChatStorage",
    # NEW OPTIMAL API (recommended)
    "ContextMiddleware",
    "create_middleware",
    "Container",
    "create_container",
    "ContextConfig",
    # OLD API (backward compatibility)
    "ContextEngineerBuilder",
    "ContextEngineerConfig",
    "Fact",
    "MessageTag",
    "RetrievalContext",
    "GlobalContextRetriever",
    "RelevantFactsRetriever",
    "ConversationSummarizer",
    "FactDeduplicator",
    "RelevanceFilter",
    "ContextFormatter",
    "MessageTagger",
    "MessageSeparator",
    "QueueBuilder",
]
