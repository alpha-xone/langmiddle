"""Fact deduplicator implementing FactProcessor protocol."""

import logging
from typing import Any

from ..core.protocols import FactProcessor
from ..models import Fact

logger = logging.getLogger(__name__)


class FactDeduplicator:
    """Removes duplicate facts using ID and content matching.

    Implements FactProcessor protocol for deduplication.
    Expects context to contain 'primary' key with primary facts.
    """

    def process(
        self,
        facts: list[Fact],
        context: dict[str, Any],
    ) -> list[Fact]:
        """Remove duplicate facts.

        Args:
            facts: Facts to deduplicate (secondary facts)
            context: Must contain 'primary' key with primary facts list

        Returns:
            Unique facts (duplicates of primary removed)
        """
        primary_facts = context.get("primary", [])
        if not primary_facts:
            logger.debug("No primary facts for deduplication, returning all facts")
            return facts

        if not facts:
            return []

        # Create lookup sets from primary facts
        primary_ids = {f.id for f in primary_facts}
        primary_contents = {self._normalize_content(f.content) for f in primary_facts}

        # Filter duplicates
        unique_facts = []
        for fact in facts:
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

        removed = len(facts) - len(unique_facts)
        if removed > 0:
            logger.info(f"Deduplication: {len(facts)} → {len(unique_facts)} facts ({removed} duplicates removed)")

        return unique_facts

    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison.

        Args:
            content: Raw content text

        Returns:
            Normalized content (lowercase, stripped)
        """
        return content.lower().strip()


# Verify protocol implementation at module load time
_instance = FactDeduplicator.__new__(FactDeduplicator)
assert isinstance(_instance, FactProcessor), "FactDeduplicator must implement FactProcessor protocol"
