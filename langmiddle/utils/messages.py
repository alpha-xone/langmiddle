"""Message utility functions for LangChain middleware.

This module provides common utilities for working with messages across
different middleware components.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from langchain.embeddings import Embeddings, init_embeddings
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.messages.utils import count_tokens_approximately

from ..utils.logging import get_graph_logger

logger = get_graph_logger(__name__)

DEFAULT_TOKEN_COUNTER = count_tokens_approximately


def embed_messages(
    embedder: Embeddings | str,
    contents: str | list[str],
    **kwargs: Any,
) -> list[list[float]] | None:
    """Embed a list of messages using the provided embedder.

    Args:
        embedder: An instance of Embeddings to use for embedding.
        messages: List of messages (either AnyMessage or dicts) to embed.

    Returns:
        List of embedding vectors, or None if embedding fails.
    """
    if isinstance(embedder, str):
        embedder = init_embeddings(embedder, **kwargs)

    if not isinstance(embedder, Embeddings):
        logger.error("Embedder is not an Embeddings instance")
        return None

    try:
        if isinstance(contents, str):
            contents = [contents]
        vectors = embedder.embed_documents(contents)
        return vectors
    except Exception:
        logger.error(
            f"Embedding failed for messages: {[content[:30] + '...' for content in contents[:5]]} ..."
        )
        return None


def is_middleware_message(msg: AnyMessage | dict) -> bool:
    """
    Check if a message is a middleware message.
    """
    if isinstance(msg, dict):
        tag = msg.get("additional_kwargs", {}).get("tag", "")
    else:
        tag = getattr(msg, "additional_kwargs", {}).get("tag", "")

    return tag.startswith("langmiddle:")


def is_tool_message(msg: AnyMessage | dict) -> bool:
    """Check if a message is a tool message.

    A message is considered a tool message if:
    1. It has type 'tool', OR
    2. It's an AI message that calls tools (finish_reason == 'tool_calls')

    This function supports both LangChain message objects and dictionary representations
    of messages, making it flexible for use across different contexts.

    Args:
        msg: Message to check. Can be either:
            - A LangChain message object (AnyMessage)
            - A dictionary with 'type' and optional 'response_metadata' keys

    Returns:
        True if message is tool-related, False otherwise.

    Examples:
        With LangChain message objects:

        >>> from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
        >>> tool_msg = ToolMessage(content="result", tool_call_id="123")
        >>> is_tool_message(tool_msg)
        True

        >>> ai_msg = AIMessage(
        ...     content="",
        ...     response_metadata={"finish_reason": "tool_calls"}
        ... )
        >>> is_tool_message(ai_msg)
        True

        >>> human_msg = HumanMessage(content="Hello")
        >>> is_tool_message(human_msg)
        False

        With dictionary representations:

        >>> tool_dict = {"type": "tool", "content": "result"}
        >>> is_tool_message(tool_dict)
        True

        >>> ai_dict = {
        ...     "type": "ai",
        ...     "response_metadata": {"finish_reason": "tool_calls"}
        ... }
        >>> is_tool_message(ai_dict)
        True

        >>> human_dict = {"type": "human", "content": "Hello"}
        >>> is_tool_message(human_dict)
        False
    """
    if isinstance(msg, dict):
        msg_type = msg.get("type")
        response = msg.get("response_metadata", {})
    else:
        msg_type = getattr(msg, "type", None)
        response = getattr(msg, "response_metadata", {})

    # Check if it's a tool message type
    if msg_type == "tool":
        return True

    # Check if it's an AI message that triggers tool calls
    if msg_type == "ai":
        finish_reason = response.get("finish_reason", "")
        return finish_reason == "tool_calls"

    return False


def filter_tool_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Filter out tool messages from a message list.

    Args:
        messages: List of messages (AnyMessage) to filter.

    Returns:
        List of messages excluding tool-related messages.

    Examples:
        >>> from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        >>> messages = [
        ...     HumanMessage(content="Hello"),
        ...     AIMessage(content="", response_metadata={"finish_reason": "tool_calls"}),
        ...     ToolMessage(content="result", tool_call_id="123"),
        ...     AIMessage(content="Done")
        ... ]
        >>> filtered = filter_tool_messages(messages)
        >>> len(filtered)
        2
        >>> filtered[0].type
        'human'
        >>> filtered[1].type
        'ai'
    """
    return [msg for msg in messages if not is_tool_message(msg)]


def split_messages(
    messages: list[AnyMessage],
    by_tags: list[str],
) -> dict[str, list[AnyMessage]]:
    """Split messages by checking additional_kwargs into tagged and regular messages.

    Args:
        messages: List of messages (either AnyMessage or dicts) to split.
        by_tags: The tags in additional_kwargs to use for splitting.
                 Tag will be in the format of {"tag": tag}.

    Returns:
        Dict with keys as tags and values as lists of messages.
    """
    tagged_msgs = defaultdict(list)
    for msg in messages:
        additional_kwargs = getattr(msg, "additional_kwargs", {})
        if any(additional_kwargs.get("tag") == tag for tag in by_tags):
            for tag in by_tags:
                if additional_kwargs.get("tag") == tag:
                    tagged_msgs[tag].append(msg)
        else:
            tagged_msgs["default"].append(msg)

    return tagged_msgs


def message_string_contents(msg: AnyMessage | dict) -> list[str]:
    """Get the content of a message.

    Args:
        msg: Message to get content from. Can be either:
            - A LangChain message object (AnyMessage)
            - A dictionary with 'content' key

    Returns:
        The content string of the message.
    """
    if isinstance(msg, dict):
        return [msg.get("content", "")]

    if isinstance(msg.content, str):
        return [msg.content]

    if isinstance(msg.content, list):
        block_strings: list[str] = [block for block in msg.content if isinstance(block, str)]
        if block_strings:
            return block_strings

    # Return empty list if content is neither str nor list of strings
    return []


def normalized_token_counts(
    messages: list[AnyMessage],
    *,
    overhead_tokens: int = 2000,
    token_counter: Callable[[list[AnyMessage]], int] = DEFAULT_TOKEN_COUNTER,
) -> list[AnyMessage]:
    """Get normalized token counts for AI messages in a message list.

    Args:
        messages: List of messages (AnyMessage) to analyze.
        overhead_tokens: Number of overhead tokens to consider in the approximation.
        token_counter: Function to count tokens approximately in a list of messages.

    Returns:
        List of messages with normalized token counts.
    """
    candidates = [8, 16, 32, 64, 128, 256, 512, 1024]
    tolerance = 0.35  # 35% tolerance

    for i, msg in enumerate(messages):
        # Skip non-AI messages - only AI messages have usage metadata
        if not isinstance(msg, AIMessage):
            continue

        # Check availability of usage token
        if not (msg.usage_metadata and isinstance(msg.usage_metadata, dict)):
            continue
        total_tokens = msg.usage_metadata.get("total_tokens", 0)
        if total_tokens <= 0:
            continue

        # Compare with approx tokens since last token count
        total_tokens = max(total_tokens, 1)
        approx_tokens = token_counter(messages[:i]) + overhead_tokens
        if not msg.response_metadata:
            msg.response_metadata = {}
        msg.response_metadata["approx_tokens"] = approx_tokens
        for scale in candidates:
            if abs(total_tokens - approx_tokens * scale) / total_tokens < tolerance:
                # Update normalized token scale and used counts
                msg.response_metadata["token_scale"] = scale
                # Update token counts in usage metadata
                for token_type in ["input", "output", "total"]:
                    if f"{token_type}_tokens" in msg.usage_metadata:
                        msg.usage_metadata[f"{token_type}_tokens"] = int(msg.usage_metadata[f"{token_type}_tokens"] // scale)
                    if f"{token_type}_tokens_details" in msg.usage_metadata:
                        for token_name, token_value in msg.usage_metadata[f"{token_type}_tokens_details"].items():
                            if isinstance(token_value, (int, float)):
                                msg.usage_metadata[f"{token_type}_tokens_details"][token_name] = int(token_value // scale)
                break

    return messages
