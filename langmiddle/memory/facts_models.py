from typing import List, Literal

from pydantic import BaseModel, Field


class FactItem(BaseModel):
    """Model to represent a single fact extracted from messages."""
    content: str = Field(
        ...,
        description="Concise and self-contained fact in the original language with subject included.",
    )
    namespace: list[str] | None = Field(
        ...,
        description="Hierarchical path for organizing the fact (e.g., ['user', 'profile'])",
    )
    intensity: float | None = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Intensity of the fact, 1.0 is strong, 0.5 is moderate, 0.0 is weak.",
    )
    confidence: float | None = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the fact, 1.0 is explicit, 0.8 is implied, ≤0.5 is tentative.",
    )
    language: str | None = Field(None, description="Language of the fact.")


class ExtractedFacts(BaseModel):
    """Model to represent an array of facts extracted from messages."""
    facts: List[FactItem] = Field(..., description="List of facts extracted from the messages.")


class CurrentItem(FactItem):
    """Model to represent a fact update."""
    id: str = Field(..., description="The ID of the fact being updated.")


class CurrentFacts(BaseModel):
    """Model to represent a list of current facts."""
    facts: List[CurrentItem] = Field(..., description="List of current facts.")


class ActionItem(CurrentItem):
    """Model to represent a fact update with an action."""
    action: Literal["ADD", "UPDATE", "DELETE", "NONE"] = Field(
        ...,
        description="Action to take on the fact: ADD, UPDATE, DELETE, or NONE.",
    )


class FactsActions(BaseModel):
    """Model to represent a list of facts updates."""
    actions: List[ActionItem] = Field(..., description="List of facts updates.")


class AtomicQueries(BaseModel):
    """Model to represent atomic query for existing facts."""
    queries: List[str] = Field(..., description="The atomic query strings to search for existing facts.")


class CuesResponse(BaseModel):
    """Response model for generated cues."""
    cues: list[str] = Field(description="List of retrieval cue questions")


class MessagesSummary(BaseModel):
    """Model to represent a summary of messages."""
    summary: str = Field(..., description="Concise summary of the provided messages.")
