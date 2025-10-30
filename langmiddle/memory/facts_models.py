from typing import List, Literal

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """Model to represent a single fact extracted from messages."""
    content: str = Field(
        ...,
        description="Concise and self-contained fact in the original language with subject included.",
    )
    namespace: list[str] = Field(
        ...,
        description="Hierarchical path for organizing the fact (e.g., ['user', 'profile'])",
    )
    intensity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Intensity of the fact, 1.0 is strong, 0.5 is moderate, 0.0 is weak.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the fact, 1.0 is explicit, 0.8 is implied, ≤0.5 is tentative.",
    )
    language: str = Field(..., description="Language of the fact.")


class Facts(BaseModel):
    """Model to represent an array of facts extracted from messages."""
    facts: List[Fact] = Field(..., description="List of facts extracted from the messages.")


class FactItem(Fact):
    """Model to represent a fact update."""
    id: str = Field(..., description="The ID of the fact being updated.")
    event: Literal["ADD", "UPDATE", "DELETE", "NONE"]


class UpdatedFacts(BaseModel):
    """Model to represent a list of facts updates."""
    facts: List[FactItem] = Field(..., description="List of facts updates.")
