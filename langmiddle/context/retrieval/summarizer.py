"""Conversation summarizer (not implementing FactRetriever as it returns text, not facts)."""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage

from ..config.defaults import SummarizationConfig

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """Generates summaries of conversation history using LLM.

    Note: This is not a FactRetriever as it returns text summaries,
    not structured facts. It's a supporting component for context injection.
    """

    def __init__(
        self,
        model: BaseChatModel,
        config: SummarizationConfig | None = None,
    ):
        """Initialize conversation summarizer.

        Args:
            model: LLM for generating summaries
            config: Optional summarization configuration
        """
        self.model = model
        self.config = config or SummarizationConfig()

    def summarize(
        self,
        messages: list[BaseMessage],
    ) -> str | None:
        """Generate conversation summary if needed.

        Args:
            messages: Full conversation history

        Returns:
            Summary text or None if not needed
        """
        # Check if summarization is needed
        if len(messages) <= self.config.summary_threshold:
            logger.debug(
                f"Skipping summarization: {len(messages)} messages <= "
                f"{self.config.summary_threshold} threshold"
            )
            return None

        # Extract messages to summarize (exclude recent ones)
        messages_to_summarize = messages[:-self.config.summary_threshold]
        if len(messages_to_summarize) < self.config.min_messages:
            logger.debug(
                f"Not enough messages to summarize: {len(messages_to_summarize)} < "
                f"{self.config.min_messages}"
            )
            return None

        # Generate summary
        try:
            prompt = self._build_prompt(messages_to_summarize)
            response = self.model.invoke([SystemMessage(content=prompt)])
            summary = self._extract_content(response)
            logger.info(f"Generated summary of {len(messages_to_summarize)} messages")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return None

    def _build_prompt(self, messages: list[BaseMessage]) -> str:
        """Build summarization prompt.

        Args:
            messages: Messages to summarize

        Returns:
            Prompt text
        """
        # Format messages for prompt
        formatted_messages = []
        for msg in messages:
            role = msg.__class__.__name__.replace("Message", "")
            content = getattr(msg, "content", str(msg))
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            formatted_messages.append(f"{role}: {content}")

        return f"""Summarize the key points from this conversation history concisely.
Focus on: decisions made, information provided, user preferences expressed, and ongoing tasks.

Conversation to summarize:
{chr(10).join(formatted_messages)}

Provide a clear, structured summary (3-5 bullet points max):"""

    def _extract_content(self, response: Any) -> str:
        """Extract text content from model response.

        Args:
            response: Model response object

        Returns:
            Text content
        """
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                return " ".join(str(item) for item in content)
            return str(content)
        return str(response)
