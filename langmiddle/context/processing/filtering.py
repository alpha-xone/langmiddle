"""Relevance filter implementing FactProcessor protocol."""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage

from ..config.defaults import ProcessingConfig
from ..core.protocols import FactProcessor
from ..models import Fact

logger = logging.getLogger(__name__)


class RelevanceFilter:
    """Filters facts for relevance using LLM assessment.

    Implements FactProcessor protocol for relevance filtering.
    Expects context to contain 'messages' and optionally 'threshold'.
    """

    def __init__(
        self,
        model: BaseChatModel,
        config: ProcessingConfig | None = None,
    ):
        """Initialize relevance filter.

        Args:
            model: LLM for relevance assessment
            config: Optional processing configuration
        """
        self.model = model
        self.config = config or ProcessingConfig()

    def process(
        self,
        facts: list[Fact],
        context: dict[str, Any],
    ) -> list[Fact]:
        """Filter facts for relevance to conversation.

        Args:
            facts: Facts to filter
            context: Must contain 'messages' key with conversation messages
                    Optional 'threshold' key for skip threshold override

        Returns:
            Filtered list of relevant facts
        """
        messages = context.get("messages", [])
        threshold = context.get("threshold", self.config.filter_threshold)

        # Skip if too few facts
        if len(facts) <= threshold:
            logger.debug(
                f"Skipping relevance filter: {len(facts)} facts <= {threshold} threshold"
            )
            return facts

        # Build prompt
        prompt = self._build_filter_prompt(facts, messages)

        # Get LLM decision
        try:
            response = self.model.invoke([SystemMessage(content=prompt)])
            relevant_indices = self._parse_response(response, len(facts))

            if relevant_indices is None:
                logger.warning("Failed to parse LLM response, keeping all facts")
                return facts

            # Filter facts
            filtered = [facts[i] for i in relevant_indices if 0 <= i < len(facts)]
            removed = len(facts) - len(filtered)
            if removed > 0:
                logger.info(
                    f"Relevance filter: {len(facts)} → {len(filtered)} facts ({removed} irrelevant removed)"
                )
            return filtered

        except Exception as e:
            logger.error(f"Failed to filter facts: {e}")
            return facts

    def _build_filter_prompt(
        self,
        facts: list[Fact],
        messages: list[BaseMessage],
    ) -> str:
        """Build filtering prompt.

        Args:
            facts: Facts to assess
            messages: Recent conversation messages

        Returns:
            Prompt text
        """
        # Extract conversation context
        context_lines = []
        for msg in messages[-5:]:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                role = msg.__class__.__name__.replace("Message", "")
                context_lines.append(f"{role}: {msg.content}")

        # Format facts
        facts_lines = []
        for i, fact in enumerate(facts, 1):
            ns_path = " > ".join(fact.namespace) if fact.namespace else "General"
            facts_lines.append(
                f"{i}. [{ns_path}] {fact.content} (confidence: {fact.confidence:.2f})"
            )

        return f"""Given the recent conversation context, determine which facts are RELEVANT.

A fact is relevant if it:
- Directly relates to topics being discussed
- Provides useful context for understanding the conversation
- Contains information that would help respond to the user

Recent Conversation:
{chr(10).join(context_lines)}

Candidate Facts:
{chr(10).join(facts_lines)}

Respond with ONLY the numbers of relevant facts, separated by commas (e.g., "1,3,5,7").
If none are relevant, respond with "NONE".
If all are relevant, respond with "ALL"."""

    def _parse_response(self, response: Any, num_facts: int) -> list[int] | None:
        """Parse LLM response into indices.

        Args:
            response: Model response object
            num_facts: Total number of facts

        Returns:
            List of 0-based indices, or None if parsing failed
        """
        # Extract content
        if hasattr(response, "content"):
            text = response.content
            if isinstance(text, list):
                text = " ".join(str(item) for item in text)
        else:
            text = str(response)

        text = text.strip().upper()

        # Handle special cases
        if text == "NONE":
            return []
        if text == "ALL":
            return list(range(num_facts))

        # Parse numbers
        try:
            indices = [int(num.strip()) - 1 for num in text.split(",")]
            return indices
        except (ValueError, IndexError):
            logger.error(f"Failed to parse relevance filter response: {text}")
            return None


# Verify protocol implementation at module load time
_instance = RelevanceFilter.__new__(RelevanceFilter)
assert isinstance(_instance, FactProcessor), "RelevanceFilter must implement FactProcessor protocol"
