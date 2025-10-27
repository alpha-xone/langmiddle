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
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime
from pydantic import BaseModel

from langmiddle.utils.logging import get_graph_logger
from langmiddle.utils.messages import filter_tool_messages

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

logger = get_graph_logger(__name__)
# Disable propagation to avoid duplicate logs
logger._logger.propagate = False

BACKENDS = {}


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
        max_tokens_before_extraction: Token threshold to trigger extraction (None = every completion).
        token_counter: Function to count tokens in messages.
        extraction_prompt: System prompt guiding the context extraction process.

    Note:
        Current implementation focuses on extraction and storage (Phase 1).
        Future versions will add context retrieval, dynamic formatting, and
        multi-backend support to complete the context engineering pipeline.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        max_tokens_before_extraction: int | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
        backend: str = "store",
        extraction_prompt: str | None = None,
        namespace_prefix: list[str] | None = None,
    ) -> None:
        """Initialize the context engineer.

        Args:
            model: LLM model for context analysis and memory extraction.
            max_tokens_before_extraction: Token threshold to trigger extraction.
                If None, extraction runs on every agent completion.
            token_counter: Function to count tokens in messages.
            extraction_prompt: Custom prompt string guiding context extraction.
        """
        super().__init__()

        if isinstance(model, str):
            try:
                model = init_chat_model(model)
            except Exception as e:
                logger.error(f"Error initializing chat model '{model}': {e}. ")

        self.model = model
        self.max_tokens_before_extraction = max_tokens_before_extraction
        self.token_counter = token_counter
        self.memory_model: type[BaseModel] | None = None
        self._llm = None

        # Ensure valid backend and model configuration
        if backend not in BACKENDS or "model" not in BACKENDS[backend]:
            logger.warning(
                f"Invalid backend: {backend}. Available choices: {list(BACKENDS.keys())}. "
                f"Using default backend 'store'."
            )
            backend = "store"

        self.backend = backend
        if extraction_prompt:
            self.extraction_prompt = extraction_prompt
        else:
            self.extraction_prompt = BACKENDS[backend].get("prompt")
        try:
            self.memory_model = cast(type[BaseModel], BACKENDS[backend].get("model"))
        except Exception as e:
            logger.error(f"Error casting memory model for backend {backend}: {e}")

        self.namespace_prefix = namespace_prefix or ["memories"]

        if isinstance(self.model, BaseChatModel) and self.memory_model:
            # Create structured output model - store as Any to avoid type issues
            self._llm: Any = self.model.with_structured_output(self.memory_model)

        if self._llm is not None:
            logger.info(
                f"Initialized middleware {self.name} with model: {self.model.__class__.__name__}, backend: {self.backend}."
            )
        else:
            logger.error(f"Initiation failed - the middleware {self.name} will be skipped during execution.")

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

        total_tokens = self.token_counter(messages)
        return total_tokens >= self.max_tokens_before_extraction

    def _extract_memories(self, messages: list[AnyMessage]) -> list[dict] | None:
        """Extract memories from messages using the LLM.

        Filters out tool messages, then invokes the LLM with the extraction prompt
        as a SystemMessage followed by the conversation messages. The LLM returns
        structured output in MemoriesExtraction format.

        Args:
            messages: List of conversation messages.

        Returns:
            Structured output containing extracted memories.
        """
        try:
            if self._llm is None or self.memory_model is None:
                logger.error("LLM or memory model not properly initialized for extraction.")
                return None

            if self._should_extract(messages) is False:
                logger.debug("Token threshold not met, skipping extraction")
                return None

            # Filter non-tool messages
            filtered_messages = filter_tool_messages(messages)
            if not filtered_messages:
                logger.debug("No non-tool messages found, skipping extraction")
                return None

            # Invoke model - result should be `self.memory_model`
            result: Any = self._llm.invoke(
                [SystemMessage(content=self.extraction_prompt)] + filtered_messages
            )

            # Type guard: ensure result is
            if not isinstance(result, self.memory_model):
                logger.warning(f"Unexpected result type: {type(result)}, expected {self.memory_model}")
                return None

            res = getattr(result, "memories", [])
            logger.info(f"Extracted {len(res)} memories from conversation")
            return res

        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return None
