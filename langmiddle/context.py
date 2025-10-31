"""Context engineering middleware for LangChain agents.

This module provides middleware for engineering enhanced context by extracting and
managing conversation memories. It wraps model calls to enrich subsequent interactions
with relevant historical context, user preferences, and accumulated insights.

The context engineering process involves:
1. Monitoring conversation flow and token thresholds
2. Extracting key memories and insights using LLM-based analysis
3. Storing memories in flexible backends (PostgreSQL, Supabase, Firebase, SQLite, ...)
4. Retrieving and formatting relevant context for future model calls

This enables agents to maintain long-term memory and personalized understanding
across multiple conversation sessions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime

from .memory.facts_prompts import DEFAULT_FACTS_EXTRACTOR, DEFAULT_FACTS_UPDATER
from .utils.logging import get_graph_logger

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

logger = get_graph_logger(__name__)
# Disable propagation to avoid duplicate logs
logger._logger.propagate = False


class ContextEngineer(AgentMiddleware[AgentState, Runtime]):
    """Context Engineer enhanced context for agents through memory extraction and management.

    This middleware wraps model calls to provide context engineering capabilities:
    - Extracts key memories and insights from conversation messages
    - Stores memories in flexible backends (PostgreSQL, Supabase, Firebase, SQLite, ...)
    - Monitors token counts to trigger extraction at appropriate intervals
    - Prepares context for future model calls with relevant historical information

    Implementation roadmap:
    - Phase 1 (Current): Memory extraction and storage vis supported backends
    - Phase 2: Context retrieval and injection into model requests
    - Phase 3: Dynamic context formatting based on relevance scoring
    - Phase 4: Multi-backend support (vector DB, custom storage adapters)
    - Phase 5: Advanced context optimization (token budgeting, semantic compression)

    Attributes:
        model: The LLM model for context analysis and memory extraction.
        embedder: Embedding model for memory representation.
        backend: Database backend to use. Currently only supports "supabase".
        extraction_prompt: System prompt guiding the facts extraction process.
        update_prompt: Custom prompt string guiding facts updating.
        max_tokens_before_extraction: Token threshold to trigger extraction (None = every completion).
        token_counter: Function to count tokens in messages.
        model_kwargs: Additional keyword arguments for model initialization.
        embedder_kwargs: Additional keyword arguments for embedder initialization.

    Note:
        Current implementation focuses on extraction and storage (Phase 1).
        Future versions will add context retrieval, dynamic formatting, and
        multi-backend support to complete the context engineering pipeline.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        embedder: str | Embeddings,
        backend: str = "supabase",
        *,
        extraction_prompt: str = DEFAULT_FACTS_EXTRACTOR,
        update_prompt: str = DEFAULT_FACTS_UPDATER,
        max_tokens_before_extraction: int | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
        model_kwargs: dict[str, Any] | None = None,
        embedder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the context engineer.

        Args:
            model: LLM model for context analysis and memory extraction.
            embedder: Embedding model for memory representation.
            backend: Database backend to use. Currently only supports "supabase".
            extraction_prompt: Custom prompt string guiding facts extraction.
            update_prompt: Custom prompt string guiding facts updating.
            max_tokens_before_extraction: Token threshold to trigger extraction.
                If None, extraction runs on every agent completion.
            token_counter: Function to count tokens in messages.
            model_kwargs: Additional keyword arguments for model initialization.
            embedder_kwargs: Additional keyword arguments for embedder initialization.
        """
        super().__init__()

        self.max_tokens_before_extraction: int | None = max_tokens_before_extraction
        self.token_counter: TokenCounter = token_counter

        # Ensure valid backend and model configuration
        if backend != "supabase":
            logger.warning(f"Invalid backend: {backend}. Using default backend 'supabase'.")
            backend = "supabase"

        self.backend: str = backend

        self.extraction_prompt = extraction_prompt
        self.update_prompt = update_prompt

        self.model: BaseChatModel | None = None
        self.embedder: Embeddings | None = None

        # Initialize LLM model
        if isinstance(model, str):
            try:
                if model_kwargs is None:
                    model_kwargs = {}
                if "temperature" not in model_kwargs:
                    model_kwargs["temperature"] = 0.1  # Keep temperature low for consistent extractions
                model = init_chat_model(model, **model_kwargs)
            except Exception as e:
                logger.error(f"Error initializing chat model '{model}': {e}.")
                return

        if isinstance(model, BaseChatModel):
            self.model = model

        # Initialize embedding model
        if isinstance(embedder, str):
            try:
                if embedder_kwargs is None:
                    embedder_kwargs = {}
                embedder = init_embeddings(embedder, **embedder_kwargs)
            except Exception as e:
                logger.error(f"Error initializing embeddings model '{embedder}': {e}.")
                return

        if isinstance(embedder, Embeddings):
            self.embedder = embedder

        if self.model is None or self.embedder is None:
            logger.error(f"Initiation failed - the middleware {self.name} will be skipped during execution.")
        else:
            logger.info(
                f"Initialized middleware {self.name} with model {self.model.__class__.__name__} / "
                f"embedder: {self.embedder.__class__.__name__} / backend: {self.backend}."
            )

    def _should_extract(self, messages: list[AnyMessage]) -> bool:
        """Determine if extraction should be triggered based on token count.

        Args:
            messages: List of conversation messages.

        Returns:
            True if extraction should run, False otherwise.
        """
        if not messages:
            return False

        if self.max_tokens_before_extraction is None:
            # Always extract if no threshold is set
            return True

        total_tokens: int = self.token_counter(messages)
        return total_tokens >= self.max_tokens_before_extraction
