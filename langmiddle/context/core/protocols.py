"""Core protocols for context engineering.

This module defines the protocol interfaces that all components must implement.
Using protocols (PEP 544) instead of abstract base classes allows for:
- Duck typing
- No inheritance required
- Easy mocking for tests
- Clear contracts
"""

from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage


@runtime_checkable
class FactRetriever(Protocol):
    """Protocol for fact retrieval strategies.

    Implementations retrieve facts from storage based on various strategies:
    - Global context (user profile, preferences)
    - Relevant facts (semantic search)
    - Summarized context (conversation summaries)
    """

    def retrieve(
        self,
        *,
        user_id: str,
        credentials: dict[str, Any],
        messages: list[BaseMessage],
        cache: dict[str, Any] | None = None,
    ) -> list["Fact"]:
        """Retrieve facts based on context.

        Args:
            user_id: User identifier
            credentials: Authentication credentials
            messages: Conversation messages for context
            cache: Optional cache for embeddings, etc.

        Returns:
            List of retrieved facts
        """
        ...


@runtime_checkable
class FactProcessor(Protocol):
    """Protocol for fact processing strategies.

    Implementations process facts in various ways:
    - Deduplication (remove duplicates)
    - Filtering (assess relevance)
    - Ranking (sort by importance)
    - Transformation (modify structure)
    """

    def process(
        self,
        facts: list["Fact"],
        context: dict[str, Any],
    ) -> list["Fact"]:
        """Process facts based on context.

        Args:
            facts: Facts to process
            context: Additional context (primary facts, messages, etc.)

        Returns:
            Processed facts
        """
        ...


@runtime_checkable
class Formatter(Protocol):
    """Protocol for formatting facts into text.

    Implementations format facts as:
    - Readable context text
    - Structured JSON
    - Markdown
    - Custom formats
    """

    def format(
        self,
        facts: list["Fact"],
        format_type: str = "general",
    ) -> str:
        """Format facts as text.

        Args:
            facts: Facts to format
            format_type: Type of formatting (global, relevant, general)

        Returns:
            Formatted text
        """
        ...


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backend implementations.

    Implementations provide:
    - Fact storage and retrieval
    - Vector similarity search
    - Transaction support
    - Authentication
    """

    def query_facts(
        self,
        *,
        credentials: dict[str, Any],
        query_embedding: list[float] | None,
        user_id: str,
        model_dimension: int | None,
        match_threshold: float,
        match_count: int,
        filter_namespaces: list[list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Query facts from storage.

        Args:
            credentials: Authentication credentials
            query_embedding: Query embedding vector (None for exact matches)
            user_id: User identifier
            model_dimension: Embedding dimension
            match_threshold: Minimum similarity threshold
            match_count: Maximum number of results
            filter_namespaces: Optional namespace filters

        Returns:
            List of fact dictionaries
        """
        ...

    def insert_facts(
        self,
        *,
        credentials: dict[str, Any],
        user_id: str,
        facts: list[dict[str, Any]],
        embeddings: list[list[float]] | None,
        model_dimension: int | None,
    ) -> dict[str, Any]:
        """Insert facts to storage.

        Args:
            credentials: Authentication credentials
            user_id: User identifier
            facts: Facts to insert
            embeddings: Optional embedding vectors
            model_dimension: Embedding dimension

        Returns:
            Result dictionary with inserted_count, fact_ids, errors
        """
        ...

    def prepare_credentials(
        self,
        *,
        user_id: str,
        auth_token: str | None = None,
    ) -> dict[str, Any]:
        """Prepare credentials for authentication.

        Args:
            user_id: User identifier
            auth_token: Optional authentication token

        Returns:
            Prepared credentials dictionary
        """
        ...

    def authenticate(self, credentials: dict[str, Any]) -> None:
        """Authenticate with backend.

        Args:
            credentials: Authentication credentials

        Raises:
            AuthenticationError: If authentication fails
        """
        ...

    def extract_user_id(self, credentials: dict[str, Any]) -> str | None:
        """Extract user ID from credentials.

        Args:
            credentials: Authentication credentials

        Returns:
            User ID if found, None otherwise
        """
        ...


@runtime_checkable
class AuthProvider(Protocol):
    """Protocol for authentication providers.

    Implementations provide:
    - User ID extraction from runtime
    - Credential preparation
    - Authentication validation
    """

    def extract(
        self,
        runtime: Any,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Extract authentication from runtime.

        Args:
            runtime: LangGraph runtime context

        Returns:
            Tuple of (user_id, credentials) or (None, None) if extraction fails
        """
        ...


@runtime_checkable
class FactExtractor(Protocol):
    """Protocol for fact extraction from conversations.

    Implementations extract:
    - Structured facts from messages
    - Key-value pairs
    - Entities and relationships
    """

    def extract(
        self,
        messages: list[BaseMessage],
    ) -> list[dict[str, Any]]:
        """Extract facts from messages.

        Args:
            messages: Conversation messages

        Returns:
            List of extracted fact dictionaries
        """
        ...


@runtime_checkable
class FactUpdater(Protocol):
    """Protocol for determining fact update actions.

    Implementations determine:
    - CREATE (new facts)
    - UPDATE (modify existing)
    - MERGE (combine duplicates)
    - DELETE (remove outdated)
    """

    def determine_actions(
        self,
        new_facts: list[dict[str, Any]],
        existing_facts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Determine actions for fact updates.

        Args:
            new_facts: Newly extracted facts
            existing_facts: Existing facts from storage

        Returns:
            List of action dictionaries with 'action' and 'fact' keys
        """
        ...


# Type alias for Fact model (imported from models.py to avoid circular imports)
Fact = Any  # Will be replaced with actual Fact type at runtime
