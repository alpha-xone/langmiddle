"""Message utility functions for LangChain middleware.

This module provides common utilities for working with messages across
different middleware components.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.embeddings import Embeddings, init_embeddings
from langchain_core.messages import AnyMessage

from ..utils.logging import get_graph_logger

logger = get_graph_logger(__name__)


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


def filter_tool_messages(messages: Sequence[AnyMessage | dict]) -> list[AnyMessage | dict]:
    """Filter out tool messages from a message list.

    Args:
        messages: List of messages (either AnyMessage or dicts) to filter.

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
