"""Context engineering middleware for LangChain agents.

This module provides middleware for engineering enhanced context by extracting and
managing conversation memories. It wraps model calls to enrich subsequent interactions
with relevant historical context, user preferences, and accumulated insights.

The context engineering process involves:
1. Monitoring conversation flow and token thresholds
2. Extracting key memories and insights using LLM-based analysis
3. Storing memories in flexible backends (Store, vector DB, custom)
4. Retrieving and formatting relevant context for future model calls

This enables agents to maintain long-term memory and personalized understanding
across multiple conversation sessions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel

from langmiddle.utils.logging import get_graph_logger
from langmiddle.utils.messages import filter_tool_messages

from .memory.store import DEFAULT_STORE_EXTRACTOR_PROMPT, MemoriesExtraction

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

logger = get_graph_logger(__name__)
# Disable propagation to avoid duplicate logs
logger._logger.propagate = False

BACKENDS = {
    "store": {"prompt": DEFAULT_STORE_EXTRACTOR_PROMPT, "model": MemoriesExtraction},
}


class ContextEngineer(AgentMiddleware[AgentState, Runtime]):
    """Engineer enhanced context for agents through memory extraction and management.

    This middleware wraps model calls to provide context engineering capabilities:
    - Extracts key memories and insights from conversation messages
    - Stores memories in flexible backends (LangGraph Store, vector DB, custom)
    - Monitors token counts to trigger extraction at appropriate intervals
    - Prepares context for future model calls with relevant historical information

    The context engineering process enhances agent interactions by maintaining
    long-term memory and personalized understanding across sessions. Extracted
    memories are stored in formats compatible with various backends (PostgresStore,
    InMemoryStore, SQLiteStore, vector databases, etc.).

    Implementation roadmap:
    - Phase 1 (Current): Memory extraction and storage via LangGraph Store
    - Phase 2: Context retrieval and injection into model requests
    - Phase 3: Dynamic context formatting based on relevance scoring
    - Phase 4: Multi-backend support (vector DB, custom storage adapters)
    - Phase 5: Advanced context optimization (token budgeting, semantic compression)

    Examples:
        Basic context engineering setup:

        >>> from langchain_openai import ChatOpenAI
        >>> from langmiddle.context import ContextEngineer
        >>>
        >>> engineer = ContextEngineer(
        ...     model=ChatOpenAI(model="gpt-4"),
        ...     namespace_prefix=["user", "123"]
        ... )

        With token-based extraction triggering:

        >>> engineer = ContextEngineer(
        ...     model="gpt-4",
        ...     namespace_prefix=["user", "123"],
        ...     max_tokens_before_extraction=4000
        ... )

        Custom extraction prompt for targeted context:

        >>> custom_prompt = '''Extract user preferences and facts.
        ... Focus on: skills, interests, and goals.'''
        >>> engineer = ContextEngineer(
        ...     model="gpt-4",
        ...     namespace_prefix=["user"],
        ...     extraction_prompt=custom_prompt
        ... )

    Attributes:
        model: The LLM model for context analysis and memory extraction.
        max_tokens_before_extraction: Token threshold to trigger extraction (None = every completion).
        token_counter: Function to count tokens in messages.
        namespace_prefix: Default namespace path prefix for organizing memories.
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
            namespace_prefix: Default namespace prefix for organizing memories.
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
        self.memory_model = None
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

        names = self.__class__.__name__, self.model.__class__.__name__
        if self._llm is not None:
            logger.info(f"Initialized middleware {names[0]} with model: {names[1]}, backend: {self.backend}.")
        else:
            logger.error(f"Initiation failed - the middleware {names[0]} will be skipped during execution.")

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

    def _extract_memories(self, messages: list[AnyMessage]) -> MemoriesExtraction:
        """Extract memories from messages using the LLM.

        Filters out tool messages, then invokes the LLM with the extraction prompt
        as a SystemMessage followed by the conversation messages. The LLM returns
        structured output in MemoriesExtraction format.

        Args:
            messages: List of conversation messages.

        Returns:
            MemoriesExtraction containing extracted memory items.
        """
        try:
            # Filter non-tool messages
            filtered_messages = filter_tool_messages(messages)

            if not filtered_messages:
                logger.debug("No non-tool messages found, skipping extraction")
                return MemoriesExtraction(memories=[])

            # Invoke model - result should be MemoriesExtraction
            result: Any = self._llm.invoke(
                [SystemMessage(content=self.extraction_prompt)] + filtered_messages
            )

            # Type guard: ensure result is MemoriesExtraction
            if not isinstance(result, MemoriesExtraction):
                logger.warning(f"Unexpected result type: {type(result)}, expected MemoriesExtraction")
                return MemoriesExtraction(memories=[])

            # Apply namespace prefix if not already set
            for memory in result.memories:
                if not memory.namespace:
                    memory.namespace = self.namespace_prefix

            logger.info(f"Extracted {len(result.memories)} memories from conversation")
            return result

        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return MemoriesExtraction(memories=[])

    def _process_memories(self, state: AgentState) -> list[Any]:
        """Extract and process memories for context engineering.

        This method analyzes conversation messages to extract key insights that
        will enhance future context. It checks token thresholds, performs LLM-based
        extraction, and formats memories for storage.

        Part of the context engineering pipeline (Phase 1: Extraction & Storage).
        Future phases will add retrieval and dynamic context formatting.

        Args:
            state: Current agent state containing conversation messages.

        Returns:
            List of extracted memory dictionaries ready for storage.
        """
        try:
            messages: list[AnyMessage] = state.get("messages", [])

            # Check if extraction should be triggered based on token count
            if not self._should_extract(messages):
                logger.debug(
                    f"Token count below threshold ({self.max_tokens_before_extraction}), "
                    "skipping memory extraction"
                )
                return []

            # Extract memories
            extraction = self._extract_memories(messages)

            if not extraction.memories:
                logger.debug("No memories extracted")
                return []

            # Return in format that can be used with Store implementations
            return [
                {
                    "namespace": tuple(m.namespace),
                    "key": m.key,
                    "value": m.value,
                    "ttl": m.ttl,
                }
                for m in extraction.memories
            ]

        except Exception as e:
            logger.error(f"Error in memory extraction middleware: {e}")
            return []

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Wrap model call with context engineering capabilities.

        Current implementation (Phase 1):
        - Executes model call through handler
        - Extracts memories from conversation post-processing
        - Stores memories in LangGraph Store backend

        Future enhancements:
        - Pre-processing: Retrieve relevant context and inject into request
        - Dynamic formatting: Adjust context based on token budgets
        - Multi-backend: Support vector DB and custom storage adapters
        - Relevance scoring: Prioritize most important context

        Args:
            request: Model request containing state and runtime.
            handler: Function to execute the actual model call.

        Returns:
            Model response after context engineering processing.
        """
        store = request.runtime.store

        res = handler(request)

        if not self._llm:
            # Logs already handled during initiation
            return res

        if self.backend == "store" and not isinstance(store, (InMemoryStore, SqliteStore, PostgresStore)):
            logger.debug(
                f"Store backend of type {type(store).__name__} does not support memory extraction, skipping. "
                "Please use one of the supported backends: InMemoryStore, SqliteStore, PostgresStore."
            )
            return res

        # Phase 1: Extract and store memories for future context
        memories = self._process_memories(request.state)
        for memory in memories:
            logger.debug(f"Storing memory: {memory}")
            if self.backend == "store" and isinstance(store, (InMemoryStore, SqliteStore, PostgresStore)):
                store.put(**memory)

        return res

    async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Asynchronous wrap model call with context engineering capabilities.

        Current implementation (Phase 1):
        - Executes model call through handler
        - Extracts memories from conversation post-processing
        - Stores memories in LangGraph Store backend

        Future enhancements:
        - Pre-processing: Retrieve relevant context and inject into request
        - Dynamic formatting: Adjust context based on token budgets
        - Multi-backend: Support vector DB and custom storage adapters
        - Relevance scoring: Prioritize most important context

        Args:
            request: Model request containing state and runtime.
            handler: Function to execute the actual model call.

        Returns:
            Model response after context engineering processing.
        """
        store = request.runtime.store

        res = await handler(request)

        if not self._llm:
            # Logs already handled during initiation
            return res

        if self.backend == "store" and not isinstance(store, (InMemoryStore, SqliteStore, PostgresStore)):
            logger.debug(
                f"Store backend of type {type(store).__name__} does not support memory extraction, skipping. "
                "Please use one of the supported backends: InMemoryStore, SqliteStore, PostgresStore."
            )
            return res

        # Phase 1: Extract and store memories for future context
        memories = self._process_memories(request.state)
        for memory in memories:
            logger.debug(f"Storing memory: {memory}")
            if self.backend == "store" and isinstance(store, (InMemoryStore, SqliteStore, PostgresStore)):
                await store.aput(**memory)

        return res
