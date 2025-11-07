"""Stateless pipelines for context injection and extraction.

This module provides pure function pipelines with no state.
All dependencies are injected and all state is passed as parameters.
"""

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from ..config.defaults import ContextConfig
from ..core.protocols import FactProcessor, FactRetriever, Formatter
from ..models import Fact, MessageTag

logger = logging.getLogger(__name__)


def separate_messages(
    messages: list[BaseMessage],
) -> tuple[dict[MessageTag, BaseMessage], list[BaseMessage]]:
    """Separate tagged context messages from regular messages.

    Pure function with no state.

    Args:
        messages: All messages in conversation

    Returns:
        Tuple of (tagged messages by tag, regular messages)
    """
    tagged: dict[MessageTag, BaseMessage] = {}
    regular: list[BaseMessage] = []

    for msg in messages:
        additional_kwargs = getattr(msg, "additional_kwargs", {})
        tag_value = additional_kwargs.get("langmiddle_tag")

        if tag_value:
            try:
                tag = MessageTag(tag_value)
                tagged[tag] = msg
            except ValueError:
                # Unknown tag, treat as regular message
                regular.append(msg)
        else:
            regular.append(msg)

    logger.debug(f"Separated: {len(tagged)} tagged, {len(regular)} regular messages")
    return tagged, regular


def build_message_queue(
    *,
    global_facts: list[Fact],
    relevant_facts: list[Fact],
    summary: str | None,
    regular_messages: list[BaseMessage],
    formatter: Formatter,
    max_recent: int,
) -> list[BaseMessage]:
    """Build message queue with injected context.

    Pure function with no state.

    Args:
        global_facts: Global context facts
        relevant_facts: Relevant facts
        summary: Optional conversation summary
        regular_messages: Regular conversation messages
        formatter: Formatter for converting facts to text
        max_recent: Maximum recent messages to include

    Returns:
        Complete message queue with context injected
    """
    queue: list[BaseMessage] = []

    # Add global context
    if global_facts:
        text = formatter.format(global_facts, "global")
        if text:
            msg = SystemMessage(
                content=text,
                additional_kwargs={"langmiddle_tag": MessageTag.GLOBAL_CONTEXT.value},
            )
            queue.append(msg)
            logger.debug(f"Added global context: {len(global_facts)} facts")

    # Add relevant facts
    if relevant_facts:
        text = formatter.format(relevant_facts, "relevant")
        if text:
            msg = SystemMessage(
                content=text,
                additional_kwargs={"langmiddle_tag": MessageTag.RELEVANT_FACTS.value},
            )
            queue.append(msg)
            logger.debug(f"Added relevant facts: {len(relevant_facts)} facts")

    # Add conversation summary
    if summary:
        msg = AIMessage(
            content=f"## Previous Conversation Summary\n{summary}",
            additional_kwargs={"langmiddle_tag": MessageTag.CONVERSATION_SUMMARY.value},
        )
        queue.append(msg)
        logger.debug("Added conversation summary")

    # Add recent messages
    recent = regular_messages[-max_recent:]
    queue.extend(recent)

    logger.info(
        f"Built queue: {len(queue)} total messages "
        f"({len(queue) - len(recent)} context, {len(recent)} conversation)"
    )

    return queue


@dataclass
class InjectionPipeline:
    """Pipeline for context injection (pure functions, no state).

    All dependencies are injected in constructor.
    All operations are pure functions with explicit parameters.
    """

    global_retriever: FactRetriever
    relevant_retriever: FactRetriever
    deduplicator: FactProcessor
    relevance_filter: FactProcessor
    formatter: Formatter
    summarizer: Any  # ConversationSummarizer (not a protocol)

    def inject(
        self,
        *,
        messages: list[BaseMessage],
        user_id: str,
        credentials: dict[str, Any],
        config: ContextConfig,
        cache: dict[str, Any] | None = None,
    ) -> list[BaseMessage]:
        """Execute injection pipeline.

        Pure function with no state.

        Args:
            messages: All conversation messages
            user_id: User identifier
            credentials: Authentication credentials
            config: Configuration (immutable)
            cache: Optional cache for embeddings

        Returns:
            Message queue with context injected
        """
        # Step 1: Separate tagged from regular messages
        _, regular = separate_messages(messages)

        # Step 2: Retrieve global context
        global_facts = self.global_retriever.retrieve(
            user_id=user_id,
            credentials=credentials,
            messages=regular,
            cache=cache,
        )

        # Step 3: Retrieve relevant facts
        relevant_facts_raw = self.relevant_retriever.retrieve(
            user_id=user_id,
            credentials=credentials,
            messages=regular[-config.retrieval.context_window:],
            cache=cache,
        )

        # Step 4: Process facts
        relevant_facts = relevant_facts_raw
        if config.processing.enable_deduplication:
            relevant_facts = self.deduplicator.process(
                relevant_facts,
                {"primary": global_facts},
            )

        if config.processing.enable_filtering:
            relevant_facts = self.relevance_filter.process(
                relevant_facts,
                {
                    "messages": regular,
                    "threshold": config.processing.filter_threshold,
                },
            )

        # Step 5: Generate summary
        summary = self.summarizer.summarize(regular)

        # Step 6: Build message queue
        return build_message_queue(
            global_facts=global_facts,
            relevant_facts=relevant_facts,
            summary=summary,
            regular_messages=regular,
            formatter=self.formatter,
            max_recent=config.max_recent_messages,
        )
