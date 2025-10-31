from typing import Any

from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate

from ..utils.logging import get_graph_logger
from ..utils.messages import filter_tool_messages
from .facts_models import CurrentFacts, ExtractedFacts, FactsUpdates
from .facts_prompts import DEFAULT_FACTS_EXTRACTOR, DEFAULT_FACTS_UPDATER

logger = get_graph_logger(__name__)


def extract_facts(
    model: BaseChatModel,
    *,
    extraction_prompt: str = DEFAULT_FACTS_EXTRACTOR,
    messages: list[AnyMessage | dict],
) -> ExtractedFacts | None:
    """
    Extract facts from a list of messages.
    """
    # Validate and set defaults
    if not isinstance(model, BaseChatModel):
        logger.error(f"Model is not a BaseChatModel: {model}")
        return None

    filtered_messages = filter_tool_messages(messages)
    if not filtered_messages:
        logger.debug("No messages to process after filtering tool messages")
        return None

    try:
        facts: Any = (
            ChatPromptTemplate.from_template(extraction_prompt)
            | model.with_structured_output(ExtractedFacts)
        ).invoke({'messages': filtered_messages})
    except Exception as e:
        logger.error(f"Error invoking model for fact extraction: {e}")
        return None

    # Type guard: ensure result is `ExtractedFacts`
    if not isinstance(facts, ExtractedFacts):
        logger.warning(f"Unexpected result type: {type(facts)}, expected {ExtractedFacts}")
        return None

    logger.info(f"Extracted {len(facts.facts)} facts from messages")
    return facts


def query_facts(
    embedder: Embeddings,
    backend: str = "supabase",
    *,
    facts: list[dict] | ExtractedFacts,
) -> CurrentFacts | None:
    """Query facts from the long-term memories."""

    if isinstance(facts, ExtractedFacts):
        new_facts: list[dict] = facts.model_dump().get("facts", [])
    else:
        new_facts = facts
    contents: list[str] = [fact.get("content", " ") for fact in new_facts]

    try:
        vectors = embedder.embed_documents(contents)
        logger.debug(f"Generated embeddings for {len(vectors)} extracted facts with dimension {len(vectors[0])}")
    except Exception as e:
        logger.error(f"Error generating embeddings for extracted facts: {e}")
        logger.error(f"Failed facts: {[content[:30] + '...' for content in contents[:5]]} ...")
        return None

    # Query from database
    if backend == "supabase":
        try:
            cur_facts = CurrentFacts(facts=[])
        except Exception as e:
            logger.error(f"Error querying facts from {backend}: {e}")
            return None
    else:
        logger.error(f"Unsupported backend: {backend}. Supported: `supabase`")
        return None

    return cur_facts


def update_facts(
    model: BaseChatModel,
    *,
    update_prompt: str = DEFAULT_FACTS_UPDATER,
    current_facts: list[dict] | CurrentFacts,
    new_facts: list[dict] | ExtractedFacts,
) -> FactsUpdates | None:
    """
    Update facts with new information from a list of messages.
    """
    # Validate and set defaults
    if not isinstance(model, BaseChatModel):
        logger.error(f"Model is not a BaseChatModel: {model}")
        return None

    try:
        updates: Any = (
            ChatPromptTemplate.from_template(update_prompt)
            | model.with_structured_output(FactsUpdates)
        ).invoke({"current_facts": current_facts, "new_facts": new_facts})
    except Exception as e:
        logger.error(f"Error invoking model for fact update: {e}")
        return None

    # Type guard: ensure result is `FactsUpdates`
    if not isinstance(updates, FactsUpdates):
        logger.warning(f"Unexpected result type: {type(updates)}, expected {FactsUpdates}")
        return None

    return updates
