"""Message queue management for context injection."""

import logging
from typing import Sequence

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage as AnyMessage
from langchain_core.messages import SystemMessage

from .models import MessageTag

logger = logging.getLogger(__name__)


class MessageTagger:
    """Manages message tagging with context metadata.

    Tags are stored in message.additional_kwargs to enable
    identification and replacement across conversation turns.
    """

    @staticmethod
    def tag_message(message: AnyMessage, tag: MessageTag) -> AnyMessage:
        """Add tag to message.

        Args:
            message: Message to tag
            tag: Tag to apply

        Returns:
            Tagged message (modified in-place)
        """
        if not hasattr(message, "additional_kwargs"):
            message.additional_kwargs = {}
        elif message.additional_kwargs is None:
            message.additional_kwargs = {}

        message.additional_kwargs["langmiddle_tag"] = tag.value
        return message

    @staticmethod
    def get_tag(message: AnyMessage) -> MessageTag | None:
        """Extract tag from message.

        Args:
            message: Message to check

        Returns:
            MessageTag if present, None otherwise
        """
        kwargs = getattr(message, "additional_kwargs", {})
        if kwargs is None:
            return None

        tag_value = kwargs.get("langmiddle_tag")
        if tag_value:
            try:
                return MessageTag(tag_value)
            except ValueError:
                logger.warning(f"Invalid message tag: {tag_value}")
                return None
        return None

    @staticmethod
    def has_tag(message: AnyMessage, tag: MessageTag) -> bool:
        """Check if message has specific tag.

        Args:
            message: Message to check
            tag: Tag to look for

        Returns:
            True if message has the tag
        """
        current_tag = MessageTagger.get_tag(message)
        return current_tag == tag


class MessageSeparator:
    """Separates tagged context messages from regular conversation messages.

    This allows replacing old context messages with updated ones
    while preserving the regular conversation flow.
    """

    def separate(
        self,
        messages: Sequence[AnyMessage],
    ) -> tuple[dict[MessageTag, AnyMessage], list[AnyMessage]]:
        """Separate messages into tagged and regular.

        Args:
            messages: All messages in the conversation

        Returns:
            Tuple of (tagged messages by tag, regular messages)
        """
        tagged: dict[MessageTag, AnyMessage] = {}
        regular: list[AnyMessage] = []

        for msg in messages:
            tag = MessageTagger.get_tag(msg)
            if tag:
                # Store latest message for each tag
                tagged[tag] = msg
                logger.debug(f"Found tagged message: {tag.value}")
            else:
                regular.append(msg)

        logger.info(
            f"Separated messages: {len(tagged)} tagged, {len(regular)} regular"
        )
        return tagged, regular


class QueueBuilder:
    """Builds the message queue with injected context.

    Constructs the final message queue by combining context messages
    and recent conversation messages in the correct order.
    """

    def __init__(self, max_recent_messages: int = 10):
        """Initialize queue builder.

        Args:
            max_recent_messages: Maximum number of recent messages to include
        """
        self.max_recent = max_recent_messages

    def build(
        self,
        global_context: str | None,
        relevant_facts: str | None,
        summary: str | None,
        regular_messages: list[AnyMessage],
    ) -> list[AnyMessage]:
        """Build message queue with context.

        Args:
            global_context: Formatted global context text
            relevant_facts: Formatted relevant facts text
            summary: Conversation summary text
            regular_messages: Regular conversation messages

        Returns:
            Complete message queue with context injected
        """
        queue = []

        # Add global context (user profile)
        if global_context:
            msg = SystemMessage(content=global_context)
            MessageTagger.tag_message(msg, MessageTag.GLOBAL_CONTEXT)
            queue.append(msg)
            logger.debug("Added global context to queue")

        # Add relevant facts
        if relevant_facts:
            msg = SystemMessage(content=relevant_facts)
            MessageTagger.tag_message(msg, MessageTag.RELEVANT_FACTS)
            queue.append(msg)
            logger.debug("Added relevant facts to queue")

        # Add conversation summary
        if summary:
            msg = AIMessage(content=f"## Previous Conversation Summary\n{summary}")
            MessageTagger.tag_message(msg, MessageTag.CONVERSATION_SUMMARY)
            queue.append(msg)
            logger.debug("Added conversation summary to queue")

        # Add recent messages
        recent = regular_messages[-self.max_recent:]
        queue.extend(recent)
        logger.info(
            f"Built queue: {len(queue)} total messages "
            f"({len(queue) - len(recent)} context, {len(recent)} conversation)"
        )

        return queue

    def should_inject(
        self,
        global_facts: list,
        relevant_facts: list,
        summary: str | None,
    ) -> bool:
        """Determine if context injection is needed.

        Args:
            global_facts: Global context facts
            relevant_facts: Relevant facts
            summary: Conversation summary

        Returns:
            True if any context is available to inject
        """
        has_context = bool(global_facts or relevant_facts or summary)
        if has_context:
            logger.debug("Context injection needed")
        else:
            logger.debug("No context to inject")
        return has_context
