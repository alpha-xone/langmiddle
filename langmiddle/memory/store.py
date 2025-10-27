from typing import Any

from pydantic import BaseModel, Field

DEFAULT_STORE_EXTRACTOR_PROMPT = """
You are an expert at extracting key insights and memories from conversations.
Analyze the conversation and extract important information that should be remembered.

Focus on:
- User preferences and interests
- Key facts about the user
- Important decisions or conclusions
- Recurring themes or patterns

For each memory:
- Assign a descriptive key (e.g., "favorite_color", "preferred_language")
- Structure the value as a dictionary with relevant fields
- Use appropriate namespace hierarchy (e.g., ["user", "profile"] or ["user", "preferences"])
"""


class MemoryItem(BaseModel):
    """A single memory item extracted from conversation.

    Attributes:
        namespace: Hierarchical path for organizing memories, e.g., ("user", "123").
        key: Unique identifier within the namespace.
        value: The memory content as a structured dictionary.
        ttl: Optional time-to-live in minutes for memory expiration.
    """

    namespace: list[str] = Field(
        description="Hierarchical path for organizing the memory (e.g., ['user', '123'])"
    )
    key: str = Field(description="Unique identifier for the memory within its namespace")
    value: dict[str, Any] = Field(description="The memory content with string keys")
    ttl: float | None = Field(default=None, description="Optional time-to-live in minutes")


class MemoriesExtraction(BaseModel):
    """Collection of extracted memories from conversation.

    Attributes:
        memories: List of memory items to be stored.
    """

    memories: list[MemoryItem] = Field(
        default_factory=list, description="List of extracted memory items"
    )
