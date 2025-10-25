"""Message utility functions for LangChain middleware.

This module provides common utilities for working with messages across
different middleware components.
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage


def is_tool_message(msg: AnyMessage) -> bool:
    """Check if a message is a tool message.

    A message is considered a tool message if:
    1. It has type 'tool', OR
    2. It's an AI message that calls tools (finish_reason == 'tool_calls')

    Args:
        msg: Message to check.

    Returns:
        True if message is tool-related, False otherwise.

    Examples:
        >>> from langchain_core.messages import ToolMessage, AIMessage
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
    """
    # Check if it's a tool message type
    if msg.type == "tool":
        return True

    # Check if it's an AI message that triggers tool calls
    if msg.type == "ai":
        finish_reason = getattr(msg, "response_metadata", {}).get("finish_reason", "")
        return finish_reason == "tool_calls"

    return False


def filter_tool_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Filter out tool messages from a message list.

    Args:
        messages: List of messages to filter.

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
