"""Memory extraction middleware for LangChain agents.

This module provides middleware for extracting memories/insights from conversation
messages after agent runs. Uses LLM-based extraction with structured output to
identify key information that should be stored for long-term memory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel, Field

from langmiddle.utils.logging import get_graph_logger
from langmiddle.utils.messages import filter_tool_messages

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

logger = get_graph_logger(__name__)
# Disable propagation to avoid duplicate logs
logger._logger.propagate = False

DEFAULT_EXTRACTOR_PROMPT = """You are an expert at extracting key insights and memories from conversations.
Analyze the conversation and extract important information that should be remembered.

Focus on:
- User preferences and interests
- Key facts about the user
- Important decisions or conclusions
- Recurring themes or patterns

For each memory:
- Assign a descriptive key (e.g., "favorite_color", "preferred_language")
- Structure the value as a dictionary with relevant fields
- Use appropriate namespace hierarchy (e.g., ["user", "profile"] or ["user", "preferences"])
"""


class MemoryItem(BaseModel):
    """A single memory item extracted from conversation.

    Attributes:
        namespace: Hierarchical path for organizing memories, e.g., ("user", "123").
        key: Unique identifier within the namespace.
        value: The memory content as a structured dictionary.
        ttl: Optional time-to-live in minutes for memory expiration.
    """

    namespace: list[str] = Field(
        description="Hierarchical path for organizing the memory (e.g., ['user', '123'])"
    )
    key: str = Field(description="Unique identifier for the memory within its namespace")
    value: dict[str, Any] = Field(description="The memory content with string keys")
    ttl: float | None = Field(default=None, description="Optional time-to-live in minutes")


class MemoriesExtraction(BaseModel):
    """Collection of extracted memories from conversation.

    Attributes:
        memories: List of memory items to be stored.
    """

    memories: list[MemoryItem] = Field(
        default_factory=list, description="List of extracted memory items"
    )


class MemoryExtractor(AgentMiddleware[AgentState, Any]):
    """Extract memories/insights from messages after agent runs.

    This middleware processes conversation messages using an LLM to identify and
    extract key information that should be stored for long-term memory. Extracted
    memories are returned in a format compatible with LangGraph Store implementations
    (PostgresStore, InMemoryStore, etc.).

    The middleware can optionally monitor token counts and only trigger extraction
    when a threshold is reached, similar to summarization middleware behavior.

    Examples:
        Basic usage with OpenAI:

        >>> from langchain_openai import ChatOpenAI
        >>> from langmiddle.memory import MemoryExtractor
        >>>
        >>> extractor = MemoryExtractor(
        ...     model=ChatOpenAI(model="gpt-4"),
        ...     namespace_prefix=["user", "123"]
        ... )

        With token-based triggering:

        >>> extractor = MemoryExtractor(
        ...     model="gpt-4",
        ...     namespace_prefix=["user", "123"],
        ...     max_tokens_before_extraction=4000,
        ...     messages_to_extract_from=20
        ... )

        Custom extraction prompt:

        >>> custom_prompt = '''Extract user preferences and facts.
        ... Focus on: skills, interests, and goals.'''
        >>> extractor = MemoryExtractor(
        ...     model="gpt-4",
        ...     namespace_prefix=["user"],
        ...     extraction_prompt=custom_prompt
        ... )

    Attributes:
        model: The LLM model for extraction (model name string or BaseChatModel instance).
        max_tokens_before_extraction: Token threshold to trigger extraction. If None, extraction runs on every agent completion.
        token_counter: Function to count tokens in messages.
        namespace_prefix: Default namespace path prefix for extracted memories.
        extraction_prompt: String prompt for memory extraction (used as SystemMessage).
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        max_tokens_before_extraction: int | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
        namespace_prefix: list[str] | None = None,
        extraction_prompt: str = DEFAULT_EXTRACTOR_PROMPT,
    ) -> None:
        """Initialize the memory extractor.

        Args:
            model: LLM model for extraction (name or instance).
            max_tokens_before_extraction: Token threshold to trigger extraction.
                If None, extraction runs on every agent completion.
            token_counter: Function to count tokens in messages.
            namespace_prefix: Default namespace prefix for memories.
            extraction_prompt: Custom prompt string for memory extraction.
        """
        super().__init__()

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = model
        self.max_tokens_before_extraction = max_tokens_before_extraction
        self.token_counter = token_counter
        self.namespace_prefix = namespace_prefix or ["memories"]
        self.extraction_prompt = extraction_prompt

        # Create structured output model - store as Any to avoid type issues
        self._llm: Any = self.model.with_structured_output(MemoriesExtraction)

    def _should_extract(self, messages: list[AnyMessage]) -> bool:
        """Determine if extraction should be triggered based on token count.

        Args:
            messages: List of conversation messages.

        Returns:
            True if extraction should run, False otherwise.
        """
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

    def after_agent(
        self, state: AgentState, runtime: Runtime[Any]  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Extract memories after agent execution completes.

        This hook is called after the agent finishes processing. It analyzes the
        conversation messages and extracts key insights to be stored.

        Args:
            state: Current agent state containing messages.
            runtime: Runtime information about the agent execution.

        Returns:
            Dictionary with extracted memories, or None if extraction failed/empty.
        """
        try:
            messages: list[AnyMessage] = state.get("messages", [])

            if not messages:
                logger.debug("No messages in state, skipping memory extraction")
                return None

            # Check if extraction should be triggered based on token count
            if not self._should_extract(messages):
                logger.debug(
                    f"Token count below threshold ({self.max_tokens_before_extraction}), "
                    "skipping memory extraction"
                )
                return None

            # Extract memories
            extraction = self._extract_memories(messages)

            if not extraction.memories:
                logger.debug("No memories extracted")
                return None

            # Return in format that can be used with Store implementations
            return {
                "memories": [
                    {
                        "namespace": tuple(m.namespace),
                        "key": m.key,
                        "value": m.value,
                        "ttl": m.ttl,
                    }
                    for m in extraction.memories
                ]
            }

        except Exception as e:
            logger.error(f"Error in memory extraction middleware: {e}")
            return None
