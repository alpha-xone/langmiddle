"""Core module for context engineering."""

from .middleware import ContextMiddleware, create_middleware
from .pipeline import InjectionPipeline
from .protocols import (
    AuthProvider,
    FactExtractor,
    FactProcessor,
    FactRetriever,
    FactUpdater,
    Formatter,
    StorageBackend,
)

__all__ = [
    # Protocols
    "FactRetriever",
    "FactProcessor",
    "Formatter",
    "StorageBackend",
    "AuthProvider",
    "FactExtractor",
    "FactUpdater",
    # Pipeline
    "InjectionPipeline",
    # Middleware (main entry point)
    "ContextMiddleware",
    "create_middleware",
]
