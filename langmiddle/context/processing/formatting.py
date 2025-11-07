"""Context formatter implementing Formatter protocol."""

import logging

from ..core.protocols import Formatter
from ..models import Fact

logger = logging.getLogger(__name__)


class ContextFormatter:
    """Formats facts into readable context text.

    Implements Formatter protocol for converting structured facts
    into human-readable text suitable for injection into conversation.
    """

    def format(
        self,
        facts: list[Fact],
        format_type: str = "general",
    ) -> str:
        """Format facts as readable context.

        Args:
            facts: Facts to format
            format_type: Type of formatting (global, relevant, or general)

        Returns:
            Formatted context text
        """
        if not facts:
            return ""

        lines = []

        # Add header based on format type
        if format_type == "global":
            lines.append("## User Profile & Preferences")
        elif format_type == "relevant":
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


# Verify protocol implementation at module load time
_instance = ContextFormatter.__new__(ContextFormatter)
assert isinstance(_instance, Formatter), "ContextFormatter must implement Formatter protocol"
