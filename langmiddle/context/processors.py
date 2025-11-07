"""Context processing strategies."""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage as AnyMessage
from langchain_core.messages import SystemMessage

from .models import Fact

logger = logging.getLogger(__name__)


class FactDeduplicator:
    """Removes duplicate facts using ID and content matching.

    This processor ensures no duplicate facts are included in the
    context, checking both unique IDs and normalized content.
    """

    def deduplicate(
        self,
        primary_facts: list[Fact],
        secondary_facts: list[Fact],
    ) -> list[Fact]:
        """Remove facts from secondary that duplicate primary.

        Args:
            primary_facts: Facts that should be preserved (e.g., global context)
            secondary_facts: Facts to deduplicate (e.g., retrieved facts)

        Returns:
            Unique facts from secondary_facts
        """
        if not secondary_facts:
            return []

        # Create lookup sets from primary facts
        primary_ids = {f.id for f in primary_facts}
        primary_contents = {self._normalize_content(f.content) for f in primary_facts}

        # Filter duplicates
        unique_facts = []
        for fact in secondary_facts:
            # Check ID duplication
            if fact.id in primary_ids:
                logger.debug(f"Removing duplicate fact by ID: {fact.id}")
                continue

            # Check content duplication
            normalized = self._normalize_content(fact.content)
            if normalized in primary_contents:
                logger.debug(f"Removing duplicate fact by content: {fact.content[:50]}...")
                continue

            unique_facts.append(fact)

        logger.info(
            f"Deduplication: {len(secondary_facts)} → {len(unique_facts)} facts "
            f"({len(secondary_facts) - len(unique_facts)} duplicates removed)"
        )
        return unique_facts

    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison.

        Args:
            content: Raw content text

        Returns:
            Normalized content (lowercase, stripped)
        """
        return content.lower().strip()


class RelevanceFilter:
    """Filters facts for relevance using LLM assessment.

    This processor uses an LLM to determine which facts are actually
    relevant to the conversation, reducing noise from high-similarity
    but irrelevant facts.
    """

    def __init__(
        self,
        model: BaseChatModel,
        skip_threshold: int = 3,
    ):
        """Initialize relevance filter.

        Args:
            model: LLM for relevance assessment
            skip_threshold: Skip filtering if fewer than this many facts
        """
        self.model = model
        self.skip_threshold = skip_threshold

    def filter(
        self,
        facts: list[Fact],
        conversation_context: list[AnyMessage],
    ) -> list[Fact]:
        """Filter facts for relevance to conversation.

        Args:
            facts: Candidate facts to filter
            conversation_context: Recent conversation messages

        Returns:
            Filtered list of relevant facts
        """
        # Skip if too few facts
        if len(facts) <= self.skip_threshold:
            logger.debug(
                f"Skipping relevance filter: {len(facts)} facts <= {self.skip_threshold} threshold"
            )
            return facts

        # Build prompt
        prompt = self._build_filter_prompt(facts, conversation_context)

        # Get LLM decision
        try:
            response = self.model.invoke([SystemMessage(content=prompt)])
            relevant_indices = self._parse_response(response, len(facts))

            if relevant_indices is None:
                logger.warning("Failed to parse LLM response, keeping all facts")
                return facts

            # Filter facts
            filtered = [facts[i] for i in relevant_indices if 0 <= i < len(facts)]
            logger.info(
                f"Relevance filter: {len(facts)} → {len(filtered)} facts "
                f"({len(facts) - len(filtered)} irrelevant removed)"
            )
            return filtered

        except Exception as e:
            logger.error(f"Failed to filter facts: {e}")
            return facts

    def _build_filter_prompt(
        self,
        facts: list[Fact],
        messages: list[AnyMessage],
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


class ContextFormatter:
    """Formats facts into readable context text.

    This formatter converts structured facts into human-readable
    text suitable for injection into the conversation context.
    """

    def format(
        self,
        facts: list[Fact],
        context_type: str = "general",
    ) -> str:
        """Format facts as readable context.

        Args:
            facts: Facts to format
            context_type: Type of context ("global", "relevant", or "general")

        Returns:
            Formatted context text
        """
        if not facts:
            return ""

        lines = []

        # Add header based on context type
        if context_type == "global":
            lines.append("## User Profile & Preferences")
        elif context_type == "relevant":
            lines.append("## Relevant Context from Previous Conversations")
        else:
            lines.append("## Context Information")

        # Group facts by namespace
        grouped = self._group_by_namespace(facts)

        # Format each group
        for namespace_path, namespace_facts in grouped.items():
            if namespace_path:
                lines.append(f"\n### {namespace_path}")

            for fact in namespace_facts:
                # Format fact with confidence
                confidence_str = f"(confidence: {fact.confidence:.2f})"
                lines.append(f"- {fact.content} {confidence_str}")

        return "\n".join(lines)

    def _group_by_namespace(
        self,
        facts: list[Fact],
    ) -> dict[str, list[Fact]]:
        """Group facts by namespace.

        Args:
            facts: Facts to group

        Returns:
            Dictionary mapping namespace paths to facts
        """
        grouped: dict[str, list[Fact]] = {}

        for fact in facts:
            # Create readable namespace path
            if fact.namespace:
                path = " > ".join(fact.namespace)
            else:
                path = "General"

            if path not in grouped:
                grouped[path] = []
            grouped[path].append(fact)

        return grouped
