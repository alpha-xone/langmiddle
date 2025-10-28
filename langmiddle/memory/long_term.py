"""Semantic and procedural memory extraction for LangChain agents.

This module provides structured extraction of semantic (factual knowledge) and
procedural (how-to knowledge) memories from conversations, following LangMem SDK
patterns for comprehensive memory management.

Semantic memories capture facts, preferences, and declarative knowledge about users,
while procedural memories capture workflows, processes, and step-by-step instructions.
"""

from typing import Any

from pydantic import BaseModel, Field

# Semantic Memory Extraction Prompt
DEFAULT_SEMANTIC_MEMORY_PROMPT = """
You are an ISTJ personality - methodical, detail-oriented, and systematic in organizing factual knowledge.

Extract semantic memories as FACTS about the user, their preferences, and declarative knowledge:
- Personal information (name, location, profession, etc.)
- Preferences and interests (favorite tools, topics, hobbies)
- Relationships and social context
- Goals and aspirations
- Domain expertise and knowledge areas
- Communication style preferences
- Past experiences and background

For each semantic memory:
- Assign a clear, descriptive key (e.g., "name", "preferred_ide", "expertise_areas")
- Structure the value with relevant fields and metadata
- Include confidence level if uncertain
- Add source context when helpful
- Use hierarchical namespace (e.g., ["user", "profile"], ["user", "preferences"])

Focus on PERSISTENT FACTS that remain true across conversations, not ephemeral details.
"""

# Procedural Memory Extraction Prompt
DEFAULT_PROCEDURAL_MEMORY_PROMPT = """
You are an ENTJ personality - strategic, efficiency-driven, and focused on optimizing processes.

Extract procedural memories as PROCESSES and step-by-step knowledge:
- Workflows and procedures the user follows
- Problem-solving strategies they prefer
- Step-by-step instructions they've shared or requested
- Common patterns in how they work
- Tools and their usage patterns
- Decision-making frameworks
- Troubleshooting approaches

For each procedural memory:
- Assign a descriptive key (e.g., "git_workflow", "debugging_process", "code_review_steps")
- Structure as ordered steps or decision trees when applicable
- Include context about when/why to use this procedure
- Note any variations or conditional logic
- Use hierarchical namespace (e.g., ["user", "workflows"], ["user", "procedures"])

Focus on REUSABLE PROCESSES that can guide future interactions, not one-time instructions.
"""

# Combined extraction prompt for both memory types
DEFAULT_COMBINED_MEMORY_PROMPT = """
You are an INTJ personality - strategic, analytical, and excellent at recognizing both factual patterns and process optimizations.

Extract both semantic and procedural memories from conversations:

SEMANTIC MEMORIES are FACTS - persistent knowledge about the user:
- Personal information, preferences, interests
- Goals, expertise, background
- Communication style preferences
- Relationships and context

PROCEDURAL MEMORIES are PROCESSES - reusable how-to knowledge:
- Workflows and procedures
- Problem-solving strategies
- Step-by-step instructions
- Common patterns and approaches
- Tool usage patterns

For each memory:
- Clearly label type: "semantic" or "procedural"
- Assign descriptive keys
- Structure values appropriately (facts vs. steps)
- Use hierarchical namespaces
- Focus on persistent, reusable information

Extract both types when present in the conversation.
"""


class SemanticMemory(BaseModel):
    """A semantic memory item - factual knowledge about the user.

    Attributes:
        namespace: Hierarchical path for organizing the memory.
        key: Unique identifier for this semantic fact.
        value: The factual content with relevant fields.
        confidence: Confidence level (0.0-1.0) if uncertain.
        source_context: Optional context about where this fact came from.
        ttl: Optional time-to-live in minutes.
    """

    namespace: list[str] = Field(
        description="Hierarchical path (e.g., ['user', 'profile'] or ['user', 'preferences'])"
    )
    key: str = Field(
        description="Unique identifier for the semantic fact (e.g., 'preferred_language', 'expertise')"
    )
    value: dict[str, Any] = Field(
        description="The factual content with structured fields"
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence level 0.0-1.0, if the fact is uncertain",
        ge=0.0,
        le=1.0,
    )
    source_context: str | None = Field(
        default=None,
        description="Optional context about where this fact originated",
    )
    ttl: float | None = Field(
        default=None,
        description="Optional time-to-live in minutes for memory expiration",
    )


class ProceduralMemory(BaseModel):
    """A procedural memory item - workflow or how-to knowledge.

    Attributes:
        namespace: Hierarchical path for organizing the memory.
        key: Unique identifier for this procedure.
        value: The procedural content with steps or decision logic.
        applies_when: Optional conditions for when to use this procedure.
        variations: Optional alternative approaches or variations.
        ttl: Optional time-to-live in minutes.
    """

    namespace: list[str] = Field(
        description="Hierarchical path (e.g., ['user', 'workflows'] or ['user', 'procedures'])"
    )
    key: str = Field(
        description="Unique identifier for the procedure (e.g., 'git_workflow', 'debugging_steps')"
    )
    value: dict[str, Any] = Field(
        description="Procedural content with steps, decision trees, or patterns"
    )
    applies_when: str | None = Field(
        default=None,
        description="Optional conditions or context for when to use this procedure",
    )
    variations: list[str] | None = Field(
        default=None,
        description="Optional alternative approaches or variations of this procedure",
    )
    ttl: float | None = Field(
        default=None,
        description="Optional time-to-live in minutes for memory expiration",
    )


class SemanticMemoriesExtraction(BaseModel):
    """Collection of extracted semantic memories.

    Attributes:
        memories: List of semantic memory items (facts and preferences).
    """

    memories: list[SemanticMemory] = Field(
        default_factory=list,
        description="List of semantic memory items extracted from conversation",
    )


class ProceduralMemoriesExtraction(BaseModel):
    """Collection of extracted procedural memories.

    Attributes:
        memories: List of procedural memory items (workflows and processes).
    """

    memories: list[ProceduralMemory] = Field(
        default_factory=list,
        description="List of procedural memory items extracted from conversation",
    )


class CombinedMemory(BaseModel):
    """A memory item that can be either semantic or procedural.

    Attributes:
        memory_type: Type of memory - "semantic" or "procedural".
        namespace: Hierarchical path for organizing the memory.
        key: Unique identifier for this memory.
        value: The memory content with appropriate structure.
        metadata: Additional metadata (confidence, source, conditions, etc.).
        ttl: Optional time-to-live in minutes.
    """

    memory_type: str = Field(
        description="Type of memory: 'semantic' (facts) or 'procedural' (processes)"
    )
    namespace: list[str] = Field(
        description="Hierarchical path for organizing the memory"
    )
    key: str = Field(
        description="Unique identifier for the memory"
    )
    value: dict[str, Any] = Field(
        description="The memory content with appropriate structure"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata (confidence, source_context, applies_when, variations, etc.)",
    )
    ttl: float | None = Field(
        default=None,
        description="Optional time-to-live in minutes for memory expiration",
    )


class CombinedMemoriesExtraction(BaseModel):
    """Collection of both semantic and procedural memories.

    Attributes:
        memories: List of memory items (both semantic and procedural).
    """

    memories: list[CombinedMemory] = Field(
        default_factory=list,
        description="List of extracted memories (both semantic and procedural)",
    )
