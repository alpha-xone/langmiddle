from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from langchain.agents.middleware.summarization import DEFAULT_SUMMARY_PROMPT
from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..utils.logging import get_graph_logger
from ..utils.messages import is_tool_message
from .facts_models import (
    AtomicQueries,
    CuesResponse,
    CurrentFacts,
    ExtractedFacts,
    FactsActions,
    MessagesSummary,
)
from .facts_prompts import (
    DEFAULT_CUES_PRODUCER,
    DEFAULT_FACTS_EXTRACTOR,
    DEFAULT_FACTS_UPDATER,
    DEFAULT_PREV_SUMMARY,
    DEFAULT_QUERY_BREAKER,
)

if TYPE_CHECKING:
    from ..storage.base import ChatStorageBackend

logger = get_graph_logger(__name__)

T = TypeVar('T')

ALWAYS_LOADED_NAMESPACES = [
    ["user", "personal_info"],
    ["user", "professional"],
    ["user", "preferences", "communication"],
    ["user", "preferences", "formatting"],
    ["user", "preferences", "topics"],
]

__all__ = [
    "ALWAYS_LOADED_NAMESPACES",
    "extract_facts",
    "get_actions",
    "apply_fact_actions",
    "query_existing_facts",
    "formatted_facts",
    "generate_fact_cues",
    "break_query_into_atomic",
]


def _invoke_structured_model(
    model: Runnable,
    prompt_template: str,
    input_vars: dict[str, Any],
    expected_type: type[T],
    function_name: str,
    fallback_value: T | None = None,
) -> T | None:
    """Common wrapper for invoking structured LLM models with validation and logging.

    Args:
        model: Runnable model with structured output already configured
        prompt_template: Prompt template string
        input_vars: Dictionary of variables to pass to the prompt
        expected_type: Expected type of the structured output
        function_name: Name of the calling function (for logging)
        fallback_value: Optional fallback value to return on error

    Returns:
        Structured output of expected_type, or fallback_value on error
    """
    if not isinstance(model, Runnable):
        logger.error(f"[{function_name}] Model is not a Runnable: {type(model)}")
        return fallback_value

    if isinstance(model, BaseChatModel):
        model = model.with_structured_output(expected_type)

    try:
        result: Any = (
            ChatPromptTemplate.from_template(prompt_template)
            | model
        ).invoke(input_vars)

        if not isinstance(result, expected_type):
            logger.warning(
                f"[{function_name}] Unexpected result type: {type(result)}, expected {expected_type.__name__}"
            )
            return fallback_value

        logger.debug(f"[{function_name}] Successfully invoked model")
        return result

    except Exception as e:
        logger.error(f"[{function_name}] Error invoking model: {e}")
        return fallback_value


def messages_summary(
    model: Runnable,
    messages: list[AnyMessage],
    prev_summary: str = "",
    *,
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
) -> str | None:
    """Summarize a list of messages"""
    if not messages:
        return None

    if not isinstance(model, Runnable):
        logger.error("[messages_summary] Model is not a Runnable")
        return None

    input_msgs = [HumanMessage(content=DEFAULT_PREV_SUMMARY.format(prev_summary=prev_summary))] if prev_summary else []

    result = _invoke_structured_model(
        model=model,
        prompt_template=summary_prompt,
        input_vars={"messages": input_msgs + messages},
        expected_type=MessagesSummary,
        function_name="messages_summary",
        fallback_value=None,
    )

    if result is None:
        return None

    if not result.summary or not result.summary.strip():
        logger.warning("[messages_summary] Generated empty summary")
        return None

    return result.summary


def formatted_facts(facts: list[dict], adaptive: bool = False) -> str:
    """Format a list of fact dictionaries into a readable string.

    Args:
        facts: List of fact dictionaries
        adaptive: If True, format based on relevance_score (high/medium/low detail)

    Returns:
        Formatted string representation of facts
    """
    if not adaptive:
        # Standard formatting
        return "\n".join(
            f"- [{' > '.join(fact['namespace'])}] {fact['content']}"
            if isinstance(fact.get("namespace"), list) and fact.get("namespace")
            else fact['content']
            for fact in facts if fact.get("content")
        )

    # Adaptive formatting based on relevance_score
    formatted_lines = []
    for fact in facts:
        if not fact.get("content"):
            continue

        relevance = fact.get("relevance_score", 0.5)
        content = fact["content"]
        namespace = fact.get("namespace", [])
        namespace_str = " > ".join(namespace) if isinstance(namespace, list) else ""

        if relevance >= 0.8:
            # High relevance: Full detail with namespace and metadata
            line = f"- [{namespace_str}] {content}"
            if fact.get("intensity"):
                line += f" (intensity: {fact['intensity']:.1f})"
        elif relevance >= 0.5:
            # Medium relevance: Compact bullet point
            line = f"- {content}"
        else:
            # Low relevance (0.3-0.5): Minimal format
            line = f"• {content[:80]}..." if len(content) > 80 else f"• {content}"

        formatted_lines.append(line)

    return "\n".join(formatted_lines)


def break_query_into_atomic(
    model: Runnable,
    user_query: str,
    *,
    query_breaker_prompt: str = DEFAULT_QUERY_BREAKER,
) -> list[str]:
    """Break a complex user query into atomic queries for better fact retrieval.

    Args:
        model: Runnable with structured output configured for AtomicQueries
        user_query: The user's query to break down
        query_breaker_prompt: Prompt template for query breaking

    Returns:
        List of atomic query strings (or original query if breaking fails)
    """
    if not user_query or not user_query.strip():
        logger.warning("[break_query_into_atomic] Empty query provided")
        return []

    result = _invoke_structured_model(
        model=model,
        prompt_template=query_breaker_prompt,
        input_vars={"user_query": user_query},
        expected_type=AtomicQueries,
        function_name="break_query_into_atomic",
        fallback_value=None,
    )

    if result is None:
        logger.debug("[break_query_into_atomic] Falling back to original query")
        return [user_query]

    if not result.queries:
        logger.debug("[break_query_into_atomic] No atomic queries generated, using original query")
        return [user_query]

    logger.debug(f"[break_query_into_atomic] Broke query into {len(result.queries)} atomic queries")
    return result.queries


def generate_fact_cues(
    model: Runnable,
    fact_content: str,
    *,
    cues_prompt: str = DEFAULT_CUES_PRODUCER,
) -> list[str]:
    """Generate retrieval cues for a fact using LLM.

    Args:
        model: Runnable with structured output configured for CuesResponse
        fact_content: The fact content to generate cues for
        cues_prompt: Prompt template for cue generation

    Returns:
        List of generated cue strings (questions that would retrieve this fact)
    """
    if not fact_content or not fact_content.strip():
        logger.warning("[generate_fact_cues] Empty fact content provided")
        return []

    result = _invoke_structured_model(
        model=model,
        prompt_template=cues_prompt,
        input_vars={"fact": fact_content},
        expected_type=CuesResponse,
        function_name="generate_fact_cues",
        fallback_value=None,
    )

    if result is None:
        return []

    logger.debug(f"[generate_fact_cues] Generated {len(result.cues)} cues for fact")
    return result.cues


def extract_facts(
    model: Runnable,
    *,
    extraction_prompt: str = DEFAULT_FACTS_EXTRACTOR,
    messages: Sequence[AnyMessage | dict],
) -> ExtractedFacts | None:
    """
    Extract facts from a list of messages.

    Args:
        model: Runnable with structured output configured for ExtractedFacts
        extraction_prompt: Prompt template for extraction
        messages: Messages to extract facts from
    """
    filtered_messages = [msg for msg in messages if not is_tool_message(msg)]
    if not filtered_messages:
        logger.debug("[extract_facts] No messages to process after filtering tool messages")
        return None

    result = _invoke_structured_model(
        model=model,
        prompt_template=extraction_prompt,
        input_vars={'messages': filtered_messages},
        expected_type=ExtractedFacts,
        function_name="extract_facts",
        fallback_value=None,
    )

    if result is not None:
        logger.info(f"[extract_facts] Extracted {len(result.facts)} facts from messages")

    return result


def query_existing_facts(
    storage_backend: "ChatStorageBackend",
    credentials: dict[str, Any],
    embedder: Embeddings,
    new_facts: list[dict],
    user_id: str,
    embeddings_cache: dict[str, list[float]] | None = None,
) -> list[dict]:
    """Query existing facts from storage using embeddings and namespace filtering.

    Efficient batch strategy:
    1. Do ONE high-similarity query across all embeddings (find duplicates)
    2. For facts without high-similarity matches, do ONE namespace-filtered query
    3. Return deduplicated results prioritizing high-similarity matches

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
        all_existing_facts = []
        seen_ids = set()
        facts_with_high_sim = set()  # Track which new facts found high-similarity matches

        # BATCH Stage 1: High similarity search for ALL facts (find duplicates)
        # This is efficient - one query per embedding but catches most duplicates
        logger.debug(f"Running high-similarity queries for {len(embeddings)} facts")
        for idx, embedding in enumerate(embeddings):
            results = storage_backend.query_facts(
                credentials=credentials,
                query_embedding=embedding,
                user_id=user_id,
                model_dimension=model_dimension,
                match_threshold=0.85,  # Very high threshold - likely duplicates
                match_count=3,  # Fewer results for high-similarity
                filter_namespaces=None,
            )

            if results:
                facts_with_high_sim.add(idx)
                for fact in results:
                    fact_id = fact.get("id")
                    if fact_id and fact_id not in seen_ids:
                        seen_ids.add(fact_id)
                        fact["_match_type"] = "high_similarity"
                        fact["_source_idx"] = idx  # Track which new fact this matches
                        all_existing_facts.append(fact)

        # BATCH Stage 2: Namespace filtering for facts WITHOUT high-similarity matches
        # Collect all namespace filters from facts that need them
        facts_needing_ns_query = [
            (idx, new_facts[idx])
            for idx in range(len(new_facts))
            if idx not in facts_with_high_sim and new_facts[idx].get("namespace")
        ]

        if facts_needing_ns_query:
            logger.debug(f"Running namespace queries for {len(facts_needing_ns_query)} facts")
            for idx, new_fact in facts_needing_ns_query:
                new_namespace = new_fact.get("namespace", [])
                if not new_namespace:
                    continue

                # Build namespace filter: include exact match and parent namespaces
                fact_namespace_filters = []
                for i in range(1, len(new_namespace) + 1):
                    fact_namespace_filters.append(new_namespace[:i])

                results = storage_backend.query_facts(
                    credentials=credentials,
                    query_embedding=embeddings[idx],
                    user_id=user_id,
                    model_dimension=model_dimension,
                    match_threshold=0.70,  # Lower threshold within same namespace
                    match_count=5,
                    filter_namespaces=fact_namespace_filters,
                )

                for fact in results:
                    fact_id = fact.get("id")
                    if fact_id and fact_id not in seen_ids:
                        seen_ids.add(fact_id)
                        fact["_match_type"] = "namespace"
                        fact["_source_idx"] = idx
                        all_existing_facts.append(fact)

        # Sort results: prioritize by match type and similarity
        # Order: high_similarity (likely duplicates) > namespace matches
        match_type_priority = {"high_similarity": 0, "namespace": 1}
        all_existing_facts.sort(
            key=lambda x: (
                match_type_priority.get(x.get("_match_type", "namespace"), 2),
                -x.get("similarity", 0),  # Then by similarity descending
            )
        )

        high_sim_count = sum(1 for f in all_existing_facts if f.get("_match_type") == "high_similarity")
        ns_count = sum(1 for f in all_existing_facts if f.get("_match_type") == "namespace")

        logger.info(
            f"Found {len(all_existing_facts)} unique existing facts in {len(embeddings) + len(facts_needing_ns_query)} queries: "
            f"{high_sim_count} high-similarity, {ns_count} namespace matches"
        )
        return all_existing_facts

    except Exception as e:
        logger.error(f"Error querying existing facts: {e}")
        return []


def get_actions(
    model: Runnable,
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

    Args:
        model: Runnable with structured output configured for FactsActions
        update_prompt: Prompt template for determining actions
        current_facts: Existing facts from storage
        new_facts: Newly extracted facts
    """
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

    logger.debug(f"[get_actions] Mapped {len(id_mapping)} fact IDs to simple numeric strings")

    result = _invoke_structured_model(
        model=model,
        prompt_template=update_prompt,
        input_vars={"current_facts": mapped_facts, "new_facts": new_facts},
        expected_type=FactsActions,
        function_name="get_actions",
        fallback_value=None,
    )

    if result is None:
        return None

    # Map simple IDs back to original IDs in the actions
    for action in result.actions:
        simple_id = action.id
        if simple_id in reverse_mapping:
            action.id = reverse_mapping[simple_id]
            logger.debug(f"[get_actions] Mapped ID {simple_id} back to {action.id}")
        else:
            logger.warning(f"[get_actions] Simple ID {simple_id} not found in reverse mapping")

    return result


def apply_fact_actions(
    storage_backend: "ChatStorageBackend",
    credentials: dict[str, Any],
    embedder: Embeddings,
    user_id: str,
    actions: list[dict],
    embeddings_cache: dict[str, list[float]] | None = None,
    model: Runnable | None = None,
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
        model: Optional Runnable with structured output for generating retrieval cues for facts

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

                # Validate generated embeddings
                if not new_embeddings or not all(new_embeddings):
                    logger.error("Failed to generate valid embeddings for ADD actions")
                    stats["errors"].append("Failed to generate embeddings for ADD actions")
                    return stats

                # Fill in the generated embeddings
                for i, embedding in enumerate(new_embeddings):
                    idx = content_indices[i]
                    add_embeddings[idx] = embedding

                    # Update cache if provided
                    if embeddings_cache is not None:
                        embeddings_cache[add_contents[idx]] = embedding

            # Validate all embeddings are present and have consistent dimensions
            if not all(add_embeddings):
                logger.error("Some ADD embeddings are missing")
                stats["errors"].append("Incomplete embeddings for ADD actions")
                return stats

            embedding_dims = [len(emb) for emb in add_embeddings if emb]
            if not embedding_dims:
                logger.error("No valid embeddings for ADD actions")
                stats["errors"].append("No valid embeddings generated")
                return stats

            if len(set(embedding_dims)) > 1:
                logger.error(f"Inconsistent embedding dimensions in ADD actions: {set(embedding_dims)}")
                stats["errors"].append(f"Inconsistent embedding dimensions: {set(embedding_dims)}")
                return stats

            model_dimension = embedding_dims[0]

            # Generate cues for all facts if model is provided
            all_cue_embeddings = []
            if model:
                logger.debug("Generating cues for ADD actions")
                for idx, action_item in enumerate(add_actions):
                    content = action_item.get("content", "")
                    cues = generate_fact_cues(model, content)

                    if cues:
                        # Generate embeddings for cues
                        cue_embeddings_for_fact = []
                        cue_texts_to_embed = []

                        for cue in cues:
                            if embeddings_cache and cue in embeddings_cache:
                                cue_embeddings_for_fact.append((cue, embeddings_cache[cue]))
                            else:
                                cue_texts_to_embed.append(cue)

                        # Generate missing cue embeddings
                        if cue_texts_to_embed:
                            try:
                                new_cue_embeddings = embedder.embed_documents(cue_texts_to_embed)
                                for cue_text, cue_emb in zip(cue_texts_to_embed, new_cue_embeddings):
                                    cue_embeddings_for_fact.append((cue_text, cue_emb))
                                    if embeddings_cache is not None:
                                        embeddings_cache[cue_text] = cue_emb
                            except Exception as e:
                                logger.error(f"Error generating cue embeddings for fact {idx}: {e}")

                        all_cue_embeddings.append(cue_embeddings_for_fact)
                        logger.debug(f"Generated {len(cue_embeddings_for_fact)} cues for fact {idx}")
                    else:
                        all_cue_embeddings.append([])
            else:
                # No model provided, no cues generated
                all_cue_embeddings = [[] for _ in add_actions]

            for idx, (action_item, embedding) in enumerate(zip(add_actions, add_embeddings)):
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
                        cue_embeddings=[all_cue_embeddings[idx]] if all_cue_embeddings else None,
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

                # Validate generated embeddings
                if not new_embeddings or not all(new_embeddings):
                    logger.error("Failed to generate valid embeddings for UPDATE actions")
                    stats["errors"].append("Failed to generate embeddings for UPDATE actions")
                    # Continue with other actions
                    for action_item in update_actions:
                        stats["skipped"] += 1
                    return stats

                # Fill in the generated embeddings
                for i, embedding in enumerate(new_embeddings):
                    idx = content_indices[i]
                    update_embeddings[idx] = embedding

                    # Update cache if provided
                    if embeddings_cache is not None:
                        embeddings_cache[update_contents[idx]] = embedding

            # Validate embeddings before updating
            if not all(update_embeddings):
                logger.error("Some UPDATE embeddings are missing")
                stats["errors"].append("Incomplete embeddings for UPDATE actions")
                stats["skipped"] += len(update_actions)
                return stats

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
