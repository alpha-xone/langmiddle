from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate

from ..utils.logging import get_graph_logger
from ..utils.messages import filter_tool_messages
from .facts_models import CurrentFacts, ExtractedFacts, FactsActions
from .facts_prompts import DEFAULT_FACTS_EXTRACTOR, DEFAULT_FACTS_UPDATER

if TYPE_CHECKING:
    from ..storage.base import ChatStorageBackend

logger = get_graph_logger(__name__)

ALWAYS_LOADED_NAMESPACES = [
    ["user", "personal_info"],
    ["user", "professional"],
    ["user", "preferences", "communication"],
    ["user", "preferences", "formatting"],
    ["user", "preferences", "topics"],
]


def extract_facts(
    model: BaseChatModel,
    *,
    extraction_prompt: str = DEFAULT_FACTS_EXTRACTOR,
    messages: Sequence[AnyMessage | dict],
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


def query_existing_facts(
    storage_backend: "ChatStorageBackend",
    credentials: dict[str, Any],
    embedder: Embeddings,
    new_facts: list[dict],
    user_id: str,
    embeddings_cache: dict[str, list[float]] | None = None,
) -> list[dict]:
    """Query existing facts from storage using embeddings and namespace filtering.

    Strategy:
    1. For each new fact, collect all unique namespace prefixes (e.g., ['user'], ['user', 'preferences'])
    2. Query with namespace filtering to find facts in related namespaces
    3. Also query without namespace filter to catch facts that might need namespace updates
    4. Return deduplicated results prioritizing namespace matches

    Args:
        storage_backend: Storage backend instance with query_facts method
        credentials: Credentials for storage backend
        embedder: Embeddings model for generating vectors
        new_facts: List of newly extracted facts
        user_id: User identifier
        embeddings_cache: Optional dict mapping content strings to pre-computed embedding vectors.
                          Only missing embeddings will be generated.

    Returns:
        List of existing relevant facts from storage
    """
    if not new_facts or embedder is None or storage_backend is None:
        return []

    try:
        # Use cached embeddings when available, generate missing ones
        contents = [fact.get("content", "") for fact in new_facts]
        embeddings = []
        contents_to_embed = []
        content_indices = []

        for idx, content in enumerate(contents):
            if embeddings_cache and content in embeddings_cache:
                embeddings.append(embeddings_cache[content])
                logger.debug(f"Using cached embedding for fact {idx}")
            else:
                contents_to_embed.append(content)
                content_indices.append(idx)
                embeddings.append(None)  # Placeholder

        # Generate embeddings for missing ones
        if contents_to_embed:
            logger.debug(f"Generating embeddings for {len(contents_to_embed)} facts")
            new_embeddings = embedder.embed_documents(contents_to_embed)

            # Fill in the generated embeddings
            for i, embedding in enumerate(new_embeddings):
                idx = content_indices[i]
                embeddings[idx] = embedding

                # Update cache if provided
                if embeddings_cache is not None:
                    embeddings_cache[contents[idx]] = embedding
        else:
            logger.debug("All embeddings found in cache")

        if not embeddings or not all(embeddings):
            logger.warning("No embeddings available for facts")
            return []

        model_dimension = len(embeddings[0])

        # Query existing facts for each embedding
        all_existing_facts = []
        seen_ids = set()

        for idx, embedding in enumerate(embeddings):
            new_fact = new_facts[idx]
            new_namespace = new_fact.get("namespace", [])

            # Strategy 1: Query with specific namespace filtering (prioritized)
            if new_namespace:
                # Build namespace filter for this specific fact:
                # Include exact match and parent namespaces
                fact_namespace_filters = []
                for i in range(1, len(new_namespace) + 1):
                    fact_namespace_filters.append(new_namespace[:i])

                results = storage_backend.query_facts(
                    credentials=credentials,
                    query_embedding=embedding,
                    user_id=user_id,
                    model_dimension=model_dimension,
                    match_threshold=0.75,  # Moderate threshold for updates
                    match_count=5,  # Get top 5 similar facts
                    filter_namespaces=fact_namespace_filters,
                )

                for fact in results:
                    fact_id = fact.get("id")
                    if fact_id and fact_id not in seen_ids:
                        seen_ids.add(fact_id)
                        fact["_namespace_match"] = True  # Mark as namespace match
                        all_existing_facts.append(fact)

            # Strategy 2: Query without namespace filter (broader search)
            # This catches facts that might be in wrong namespaces or need updating
            results = storage_backend.query_facts(
                credentials=credentials,
                query_embedding=embedding,
                user_id=user_id,
                model_dimension=model_dimension,
                match_threshold=0.80,  # Slightly higher threshold for non-namespace matches
                match_count=3,  # Fewer results without namespace filter
                filter_namespaces=None,
            )

            for fact in results:
                fact_id = fact.get("id")
                if fact_id and fact_id not in seen_ids:
                    seen_ids.add(fact_id)
                    fact["_namespace_match"] = False  # Mark as non-namespace match
                    all_existing_facts.append(fact)

        # Sort results: namespace matches first, then by similarity
        all_existing_facts.sort(
            key=lambda x: (
                not x.get("_namespace_match", False),  # Namespace matches first (False < True)
                -x.get("similarity", 0),  # Then by similarity descending
            )
        )

        logger.debug(
            f"Found {len(all_existing_facts)} unique existing facts "
            f"({sum(1 for f in all_existing_facts if f.get('_namespace_match'))} with namespace match)"
        )
        return all_existing_facts

    except Exception as e:
        logger.error(f"Error querying existing facts: {e}")
        return []


def get_actions(
    model: BaseChatModel,
    *,
    update_prompt: str = DEFAULT_FACTS_UPDATER,
    current_facts: list[dict] | CurrentFacts,
    new_facts: list[dict] | ExtractedFacts,
) -> FactsActions | None:
    """
    Update facts with new information from a list of messages.

    This function maps existing fact IDs to simple numeric strings (1, 2, 3, ...)
    before sending to the AI model to reduce errors, then maps them back to
    original IDs in the returned actions.
    """
    # Validate and set defaults
    if not isinstance(model, BaseChatModel):
        logger.error(f"Model is not a BaseChatModel: {model}")
        return None

    # Create ID mapping: original_id -> simple_id ("1", "2", "3", ...)
    id_mapping = {}
    reverse_mapping = {}

    # Extract facts list from CurrentFacts model if needed
    facts_list = current_facts.facts if isinstance(current_facts, CurrentFacts) else current_facts

    # Map existing fact IDs to simple numeric strings
    mapped_facts = []
    for idx, fact in enumerate(facts_list, start=1):
        simple_id = str(idx)

        # Get original ID from dict or Pydantic model
        if isinstance(fact, dict):
            original_id = fact.get("id")
            mapped_fact = fact.copy()
            mapped_fact["id"] = simple_id
        else:
            # Handle Pydantic model
            original_id = fact.id
            mapped_fact = fact.model_dump()
            mapped_fact["id"] = simple_id

        if original_id:
            id_mapping[original_id] = simple_id
            reverse_mapping[simple_id] = original_id

        mapped_facts.append(mapped_fact)

    logger.debug(f"Mapped {len(id_mapping)} fact IDs to simple numeric strings")

    try:
        updates: Any = (
            ChatPromptTemplate.from_template(update_prompt)
            | model.with_structured_output(FactsActions)
        ).invoke({"current_facts": mapped_facts, "new_facts": new_facts})
    except Exception as e:
        logger.error(f"Error invoking model for fact update: {e}")
        return None

    # Type guard: ensure result is `FactsActions`
    if not isinstance(updates, FactsActions):
        logger.warning(f"Unexpected result type: {type(updates)}, expected {FactsActions}")
        return None

    # Map simple IDs back to original IDs in the actions
    for action in updates.actions:
        simple_id = action.id
        if simple_id in reverse_mapping:
            action.id = reverse_mapping[simple_id]
            logger.debug(f"Mapped ID {simple_id} back to {action.id}")
        else:
            logger.warning(f"Simple ID {simple_id} not found in reverse mapping")

    return updates


def apply_fact_actions(
    storage_backend: "ChatStorageBackend",
    credentials: dict[str, Any],
    embedder: Embeddings,
    user_id: str,
    actions: list[dict],
    embeddings_cache: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Apply the determined actions to the storage backend.

    Args:
        storage_backend: Storage backend instance with insert/update/delete methods
        credentials: Credentials for storage backend
        embedder: Embeddings model for generating vectors
        user_id: User identifier
        actions: List of action dictionaries with 'action' field (ADD, UPDATE, DELETE, NONE)
        embeddings_cache: Optional dict mapping content strings to pre-computed embedding vectors.
                         Only missing embeddings will be generated.

    Returns:
        Dict with statistics: {'added': int, 'updated': int, 'deleted': int, 'errors': list}
    """
    if storage_backend is None or embedder is None:
        logger.error("Storage or embedder not initialized")
        return {
            "added": 0,
            "updated": 0,
            "deleted": 0,
            "errors": ["Storage not initialized"]
        }

    stats = {"added": 0, "updated": 0, "deleted": 0, "errors": []}

    # Separate actions by type and collect contents for batch embedding
    add_actions = []
    update_actions = []
    delete_actions = []

    for action_item in actions:
        action = action_item.get("action")
        if action == "ADD":
            add_actions.append(action_item)
        elif action == "UPDATE":
            update_actions.append(action_item)
        elif action == "DELETE":
            delete_actions.append(action_item)

    # Batch generate embeddings for ADD actions (use cache when available)
    if add_actions:
        try:
            add_contents = [a.get("content", "") for a in add_actions]
            add_embeddings = []
            contents_to_embed = []
            content_indices = []

            # Check cache for existing embeddings
            for idx, content in enumerate(add_contents):
                if embeddings_cache and content in embeddings_cache:
                    add_embeddings.append(embeddings_cache[content])
                    logger.debug(f"Using cached embedding for ADD action {idx}")
                else:
                    contents_to_embed.append(content)
                    content_indices.append(idx)
                    add_embeddings.append(None)  # Placeholder

            # Generate embeddings for missing ones
            if contents_to_embed:
                logger.debug(f"Generating embeddings for {len(contents_to_embed)} ADD actions")
                new_embeddings = embedder.embed_documents(contents_to_embed)

                # Fill in the generated embeddings
                for i, embedding in enumerate(new_embeddings):
                    idx = content_indices[i]
                    add_embeddings[idx] = embedding

                    # Update cache if provided
                    if embeddings_cache is not None:
                        embeddings_cache[add_contents[idx]] = embedding

            model_dimension = len(add_embeddings[0]) if add_embeddings else 1536

            for action_item, embedding in zip(add_actions, add_embeddings):
                try:
                    fact = {
                        "content": action_item.get("content"),
                        "namespace": action_item.get("namespace", []),
                        "language": action_item.get("language", "en"),
                        "intensity": action_item.get("intensity"),
                        "confidence": action_item.get("confidence"),
                    }

                    result = storage_backend.insert_facts(
                        credentials=credentials,
                        user_id=user_id,
                        facts=[fact],
                        embeddings=[embedding],
                        model_dimension=model_dimension,
                    )

                    if result.get("inserted_count", 0) > 0:
                        stats["added"] += 1
                    else:
                        stats["errors"].extend(result.get("errors", []))

                except Exception as e:
                    logger.error(f"Error adding fact: {e}")
                    stats["errors"].append(f"Error adding fact: {str(e)}")

        except Exception as e:
            logger.error(f"Error generating embeddings for ADD actions: {e}")
            stats["errors"].append(f"Batch embedding failed for ADD: {str(e)}")

    # Batch generate embeddings for UPDATE actions (use cache when available)
    if update_actions:
        try:
            update_contents = [a.get("content", "") for a in update_actions]
            update_embeddings = []
            contents_to_embed = []
            content_indices = []

            # Check cache for existing embeddings
            for idx, content in enumerate(update_contents):
                if embeddings_cache and content in embeddings_cache:
                    update_embeddings.append(embeddings_cache[content])
                    logger.debug(f"Using cached embedding for UPDATE action {idx}")
                else:
                    contents_to_embed.append(content)
                    content_indices.append(idx)
                    update_embeddings.append(None)  # Placeholder

            # Generate embeddings for missing ones
            if contents_to_embed:
                logger.debug(f"Generating embeddings for {len(contents_to_embed)} UPDATE actions")
                new_embeddings = embedder.embed_documents(contents_to_embed)

                # Fill in the generated embeddings
                for i, embedding in enumerate(new_embeddings):
                    idx = content_indices[i]
                    update_embeddings[idx] = embedding

                    # Update cache if provided
                    if embeddings_cache is not None:
                        embeddings_cache[update_contents[idx]] = embedding

            for action_item, embedding in zip(update_actions, update_embeddings):
                try:
                    fact_id = action_item.get("id")
                    updates = {
                        "content": action_item.get("content"),
                        "intensity": action_item.get("intensity"),
                        "confidence": action_item.get("confidence"),
                        "updated_at": "now()",
                    }

                    success = storage_backend.update_fact(
                        credentials=credentials,
                        fact_id=fact_id,
                        user_id=user_id,
                        updates=updates,
                        embedding=embedding,
                    )

                    if success:
                        stats["updated"] += 1
                    else:
                        stats["errors"].append(f"Failed to update fact {fact_id}")

                except Exception as e:
                    logger.error(f"Error updating fact {action_item.get('id')}: {e}")
                    stats["errors"].append(f"Error updating fact: {str(e)}")

        except Exception as e:
            logger.error(f"Error generating embeddings for UPDATE actions: {e}")
            stats["errors"].append(f"Batch embedding failed for UPDATE: {str(e)}")

    # Process DELETE actions (no embeddings needed)
    for action_item in delete_actions:
        try:
            fact_id = action_item.get("id")
            success = storage_backend.delete_fact(
                credentials=credentials,
                fact_id=fact_id,
                user_id=user_id,
            )

            if success:
                stats["deleted"] += 1
            else:
                stats["errors"].append(f"Failed to delete fact {fact_id}")

        except Exception as e:
            logger.error(f"Error deleting fact {action_item.get('id')}: {e}")
            stats["errors"].append(f"Error deleting fact: {str(e)}")

    return stats
