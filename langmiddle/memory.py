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
    AIMessage,
    AnyMessage,
    HumanMessage,
    MessageLikeRepresentation,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
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

        Custom prompt template:

        >>> custom_prompt = ChatPromptTemplate.from_messages([
        ...     ("system", "Extract key facts about the user from this conversation."),
        ...     ("user", "{messages}")
        ... ])
        >>> extractor = MemoryExtractor(
        ...     model="gpt-4",
        ...     namespace_prefix=["user"],
        ...     prompt_template=custom_prompt
        ... )

    Attributes:
        model: The LLM model for extraction (model name string or BaseChatModel instance).
        max_tokens_before_extraction: Token threshold to trigger extraction. If None, extraction runs on every agent completion.
        messages_to_extract_from: Number of recent messages to analyze for extraction.
        token_counter: Function to count tokens in messages.
        namespace_prefix: Default namespace path prefix for extracted memories.
        prompt_template: Optional custom prompt for memory extraction.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        max_tokens_before_extraction: int | None = None,
        messages_to_extract_from: int = 20,
        token_counter: TokenCounter = count_tokens_approximately,
        namespace_prefix: list[str] | None = None,
        prompt_template: ChatPromptTemplate | None = None,
    ) -> None:
        """Initialize the memory extractor.

        Args:
            model: LLM model for extraction (name or instance).
            max_tokens_before_extraction: Token threshold to trigger extraction.
                If None, extraction runs on every agent completion.
            messages_to_extract_from: Number of recent messages to analyze.
            token_counter: Function to count tokens in messages.
            namespace_prefix: Default namespace prefix for memories.
            prompt_template: Optional custom prompt template.
        """
        super().__init__()

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = model
        self.max_tokens_before_extraction = max_tokens_before_extraction
        self.messages_to_extract_from = messages_to_extract_from
        self.token_counter = token_counter
        self.namespace_prefix = namespace_prefix or ["memories"]
        self.prompt_template = prompt_template or self._create_default_prompt()

        # Create structured output model - store as Any to avoid type issues
        self._structured_model: Any = self.model.with_structured_output(MemoriesExtraction)

    def _create_default_prompt(self) -> ChatPromptTemplate:
        """Create the default prompt template for memory extraction.

        Returns:
            A chat prompt template with system and user messages.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", DEFAULT_EXTRACTOR_PROMPT),
                ("user", "Extract memories from this conversation:\n\n{messages}"),
            ]
        )

    def _get_recent_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Get the most recent messages to analyze for extraction.

        Args:
            messages: Full list of conversation messages.

        Returns:
            List of recent messages up to messages_to_extract_from limit.
        """
        if len(messages) <= self.messages_to_extract_from:
            return messages
        return messages[-self.messages_to_extract_from:]

    def _format_messages_for_prompt(self, messages: list[AnyMessage]) -> str:
        """Format messages into a readable string for the prompt.

        Args:
            messages: List of messages to format.

        Returns:
            Formatted string representation of messages.
        """
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"AI: {msg.content}")
            else:
                formatted.append(f"{msg.__class__.__name__}: {msg.content}")
        return "\n".join(formatted)

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

            # Format messages for prompt
            formatted_messages = self._format_messages_for_prompt(filtered_messages)

            # Create prompt
            prompt = self.prompt_template.format_messages(messages=formatted_messages)

            # Invoke model - result should be MemoriesExtraction
            result: Any = self._structured_model.invoke(prompt)

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

            # Get recent messages to analyze
            recent_messages = self._get_recent_messages(messages)

            # Extract memories
            extraction = self._extract_memories(recent_messages)

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
