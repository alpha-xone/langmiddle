"""Memory extraction and management for LangChain agents.

This module provides a flexible system for extracting, storing, and retrieving
memories from agent conversations. It supports multiple storage backends and
extraction strategies.
"""

from store import MemoriesExtraction, DEFAULT_STORE_EXTRACTOR_PROMPT


__all__ = [
    "MemoriesExtraction",
    "DEFAULT_STORE_EXTRACTOR_PROMPT",
]
