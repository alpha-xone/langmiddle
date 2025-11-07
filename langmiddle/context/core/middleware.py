"""Stateless middleware for context management.

This module provides a thin orchestrator with zero state.
All dependencies are injected and all state is passed as parameters.
"""

import logging
from typing import Any

from langchain_core.messages import BaseMessage

from ..config.defaults import ContextConfig
from ..di.container import Container

logger = logging.getLogger(__name__)


class ContextMiddleware:
    """Stateless middleware for context injection.

    Zero instance variables except dependencies.
    All state passed as parameters, no side effects.

    This is the main entry point for the optimal architecture.
    """

    def __init__(self, container: Container):
        """Initialize middleware with DI container.

        Args:
            container: DI container with all dependencies
        """
        self._container = container
        self._pipeline = container.injection_pipeline()

        logger.info(
            f"Middleware initialized with {type(container.storage).__name__} storage"
        )

    @property
    def config(self) -> ContextConfig:
        """Get immutable configuration."""
        return self._container.config

    def inject_context(
        self,
        *,
        messages: list[BaseMessage],
        user_id: str,
        credentials: dict[str, Any],
        cache: dict[str, Any] | None = None,
    ) -> list[BaseMessage]:
        """Inject context into conversation messages.

        Pure orchestration with no state.

        Args:
            messages: Conversation messages
            user_id: User identifier
            credentials: Authentication credentials
            cache: Optional cache for embeddings

        Returns:
            Message queue with context injected

        Example:
            >>> middleware = ContextMiddleware(container)
            >>> result = middleware.inject_context(
            ...     messages=conversation,
            ...     user_id="user123",
            ...     credentials={"supabase_auth": token},
            ... )
        """
        logger.debug(
            f"Injecting context: {len(messages)} messages, user={user_id[:8]}..."
        )

        try:
            result = self._pipeline.inject(
                messages=messages,
                user_id=user_id,
                credentials=credentials,
                config=self.config,
                cache=cache,
            )

            logger.info(
                f"Context injection complete: {len(messages)} -> {len(result)} messages"
            )

            return result

        except Exception as e:
            logger.error(f"Context injection failed: {e}", exc_info=True)
            raise


def create_middleware(
    *,
    storage: Any,  # Must implement StorageBackend protocol
    embedder: Any | None = None,
    model: Any | None = None,
    config: ContextConfig | None = None,
) -> ContextMiddleware:
    """Factory function for creating middleware.

    This is the main entry point for users of the optimal architecture.

    Args:
        storage: Storage backend (must implement StorageBackend protocol)
        embedder: Optional embeddings model (Embeddings)
        model: Optional LLM model (BaseChatModel)
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Configured middleware ready to use

    Example:
        >>> from langmiddle.storage import SupabaseBackend
        >>> from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        >>> from langmiddle.context import create_middleware, ContextConfig
        >>>
        >>> # Minimal setup (uses all defaults)
        >>> storage = SupabaseBackend(url="...", key="...")
        >>> embedder = OpenAIEmbeddings()
        >>> model = ChatOpenAI()
        >>> middleware = create_middleware(storage=storage, embedder=embedder, model=model)
        >>>
        >>> # Custom configuration
        >>> config = ContextConfig.create(
        ...     max_recent_messages=15,
        ... )
        >>> middleware = create_middleware(
        ...     storage=storage,
        ...     embedder=embedder,
        ...     model=model,
        ...     config=config,
        ... )
        >>>
        >>> # Inject context
        >>> result = middleware.inject_context(
        ...     messages=conversation,
        ...     user_id="user123",
        ...     credentials={"supabase_auth": token},
        ... )
    """
    container = Container(storage=storage, config=config or ContextConfig.create(), embedder=embedder, model=model)
    return ContextMiddleware(container)
