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
from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime
from pydantic import BaseModel

from langmiddle.utils.logging import get_graph_logger
from langmiddle.utils.messages import filter_tool_messages

from .memory.facts_prompts import DEFAULT_FACTS_EXTRACTOR, DEFAULT_FACTS_UPDATER
from .memory.facts_models import Facts, UpdatedFacts

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
        backend: Database backend to use. Currently only supports "supabase".
        extraction_prompt: System prompt guiding the facts extraction process.
        extraction_model: Model class for facts extraction.
        update_prompt: Custom prompt string guiding facts updating.
        update_model: Custom model class for facts updating.
        max_tokens_before_extraction: Token threshold to trigger extraction (None = every completion).
        token_counter: Function to count tokens in messages.

    Note:
        Current implementation focuses on extraction and storage (Phase 1).
        Future versions will add context retrieval, dynamic formatting, and
        multi-backend support to complete the context engineering pipeline.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        backend: str = "supabase",
        *,
        extraction_prompt: str | None = None,
        extraction_model: type[BaseModel] | None = None,
        update_prompt: str | None = None,
        update_model: type[BaseModel] | None = None,
        max_tokens_before_extraction: int | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
    ) -> None:
        """Initialize the context engineer.

        Args:
            model: LLM model for context analysis and memory extraction.
            backend: Database backend to use. Currently only supports "supabase".
            extraction_prompt: Custom prompt string guiding facts extraction.
            extraction_model: Custom model class for facts extraction.
            update_prompt: Custom prompt string guiding facts updating.
            update_model: Custom model class for facts updating.
            max_tokens_before_extraction: Token threshold to trigger extraction.
                If None, extraction runs on every agent completion.
            token_counter: Function to count tokens in messages.
        """
        super().__init__()

        self.max_tokens_before_extraction = max_tokens_before_extraction
        self.token_counter = token_counter

        # Ensure valid backend and model configuration
        if backend != "supabase":
            logger.warning(
                f"Invalid backend: {backend}. Using default backend 'supabase'."
            )
            backend = "supabase"

        self.backend = backend

        if extraction_prompt:
            self.extraction_prompt = extraction_prompt
        else:
            self.extraction_prompt = DEFAULT_FACTS_EXTRACTOR

        if extraction_model is not None:
            self.facts_model = extraction_model
        else:
            self.facts_model = Facts

        if update_prompt:
            self.update_prompt = update_prompt
        else:
            self.update_prompt = DEFAULT_FACTS_UPDATER

        if update_model is not None:
            self.update_model = update_model
        else:
            self.update_model = UpdatedFacts

        if isinstance(model, str):
            try:
                model = init_chat_model(model, temperature=0.1)
            except Exception as e:
                logger.error(f"Error initializing chat model '{model}': {e}.")
                logger.error(f"Initiation failed - the middleware {self.name} will be skipped during execution.")
                return

        self.model: BaseChatModel = model
        logger.info(
            f"Initialized middleware {self.name} with model: {self.model.__class__.__name__} / backend: {self.backend}."
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

        total_tokens = self.token_counter(messages)
        return total_tokens >= self.max_tokens_before_extraction

    def _extract_facts(self, messages: list[AnyMessage]) -> list[dict] | None:
        """Extract facts from messages using the LLM.

        Filters out tool messages, then invokes the LLM with the extraction prompt
        as a SystemMessage followed by the conversation messages. The LLM returns
        structured output in given format - default `Facts`.

        Args:
            messages: List of conversation messages.

        Returns:
            Structured output containing extracted facts.
        """
        try:
            # Filter non-tool messages
            filtered_messages = filter_tool_messages(messages)
            if not filtered_messages:
                logger.debug("No non-tool messages found, skipping extraction")
                return None

            if self._should_extract(filtered_messages) is False:
                logger.debug("Token threshold not met, skipping extraction")
                return None

            # Invoke model - result should be `self.facts_model`
            result: Any = (
                ChatPromptTemplate.from_template(self.extraction_prompt)
                | self.model.with_structured_output(self.facts_model)
            ).invoke({'messages': filtered_messages})

            # Type guard: ensure result is
            if not isinstance(result, self.facts_model):
                logger.warning(f"Unexpected result type: {type(result)}, expected {self.facts_model}")
                return None

            res = getattr(result, "facts", [])
            logger.info(f"Extracted {len(res)} facts from conversation")
            return res

        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return None
