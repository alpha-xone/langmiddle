"""Context engineering middleware for LangChain agents.

This module provides middleware for engineering enhanced context by extracting and
managing conversation memories. It wraps model calls to enrich subsequent interactions
with relevant historical context, user preferences, and accumulated insights.

The context engineering process involves:
1. Monitoring conversation flow and token thresholds
2. Extracting key memories and insights using LLM-based analysis
3. Storing memories in flexible backends (PostgreSQL, Supabase, Firebase, SQLite, ...)
4. Retrieving and formatting relevant context for future model calls

This enables agents to maintain long-term memory and personalized understanding
across multiple conversation sessions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.summarization import DEFAULT_SUMMARY_PROMPT
from langchain.chat_models import init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from langchain.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    MessageLikeRepresentation,
    RemoveMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from .memory.facts_manager import (
    ALWAYS_LOADED_NAMESPACES,
    apply_fact_actions,
    break_query_into_atomic,
    extract_facts,
    formatted_facts,
    get_actions,
    messages_summary,
    query_existing_facts,
)
from .memory.facts_models import ExtractedFacts, FactsActions
from .memory.facts_prompts import (
    DEFAULT_BASIC_INFO_INJECTOR,
    DEFAULT_FACTS_EXTRACTOR,
    DEFAULT_FACTS_INJECTOR,
    DEFAULT_FACTS_UPDATER,
)
from .storage import ChatStorage
from .utils.logging import get_graph_logger
from .utils.messages import message_string_contents, split_messages
from .utils.runtime import auth_storage, get_user_id


# Type protocols for better type safety
class TokenCounter(Protocol):
    """Protocol for token counting strategies."""

    def __call__(self, messages: Iterable[MessageLikeRepresentation]) -> int:
        """Count tokens in messages."""
        ...


# Configuration dataclasses
@dataclass
class ExtractionConfig:
    """Configuration for fact extraction behavior."""

    interval: int = 3
    """Extract facts every N agent completions."""

    max_tokens: int | None = None
    """Token threshold to trigger extraction (overrides interval if set)."""

    prompt: str = DEFAULT_FACTS_EXTRACTOR
    """Prompt template for extracting facts."""

    update_prompt: str = DEFAULT_FACTS_UPDATER
    """Prompt template for updating existing facts."""


@dataclass
class SummarizationConfig:
    """Configuration for conversation summarization behavior."""

    max_tokens: int = 8000
    """Token threshold to trigger summarization."""

    keep_ratio: float = 0.5
    """Ratio of recent messages to keep after summarization (0.5 = keep last 50%)."""

    prompt: str = DEFAULT_SUMMARY_PROMPT
    """Prompt template for generating summaries."""

    prefix: str = "## Previous Conversation Summary\n"
    """Prefix to add before the summary content."""


@dataclass
class ContextConfig:
    """Configuration for context injection behavior."""

    core_namespaces: list[list[str]] = field(default_factory=lambda: ALWAYS_LOADED_NAMESPACES)
    """List of namespaces to always load into context."""

    core_prompt: str = DEFAULT_BASIC_INFO_INJECTOR
    """Prompt template for core facts injection."""

    memory_prompt: str = DEFAULT_FACTS_INJECTOR
    """Prompt template for context-specific facts injection."""


@dataclass
class _MiddlewareState:
    """Internal state tracking for the middleware (private)."""

    turn_count: int = 0
    """Number of agent turns processed."""

    extraction_count: int = 0
    """Number of fact extractions performed."""

    user_id: str = ""
    """Current user identifier."""

    core_facts: list[dict[str, Any]] = field(default_factory=list)
    """Cached core facts (loaded once per session)."""

    current_facts: list[dict[str, Any]] = field(default_factory=list)
    """Context-specific facts for current conversation."""

    embeddings_cache: dict[str, list[float]] = field(default_factory=dict)
    """Cache for reusing embeddings to improve performance."""

    summerized_msg_ids: set[str] = field(default_factory=set)
    """Set of message IDs that have been summerized."""

    def reset_session_state(self) -> None:
        """Reset per-session state while keeping caches."""
        self.turn_count = 0
        self.current_facts.clear()


load_dotenv()

logger = get_graph_logger(__name__)
# Disable propagation to avoid duplicate logs
logger._logger.propagate = False

CONTEXT_TAG = "langmiddle/context"
SUMMARY_TAG = "langmiddle/summary"
LOGS_KEY = "langmiddle:context:trace"


def _validate_storage_and_auth(
    storage: Any,
    runtime: Runtime[Any],
    backend: str,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Validate storage initialization and authenticate user.

    Args:
        storage: Storage backend instance
        runtime: Runtime context
        backend: Backend type name

    Returns:
        Tuple of (user_id, credentials, error_message)
        If successful: (user_id, credentials, None)
        If failed: (None, None, error_message)
    """
    if storage is None:
        return None, None, "Storage not initialized"

    # Get user ID
    user_id = get_user_id(
        runtime=runtime,
        backend=backend,
        storage_backend=storage.backend,
    )
    if not user_id:
        return None, None, "Missing user_id in context"

    # Authenticate and get credentials
    auth_status = auth_storage(
        runtime=runtime,
        backend=backend,
        storage_backend=storage.backend,
    )
    if "error" in auth_status:
        return None, None, f"Authentication failed: {auth_status['error']}"

    credentials = auth_status.get("credentials", {})
    return user_id, credentials, None


def _query_facts_with_validation(
    storage: Any,
    embedder: Embeddings | None,
    model: BaseChatModel | None,
    credentials: dict[str, Any],
    query_type: str,
    *,
    filter_namespaces: list[list[str]] | None = None,
    match_count: int | None = None,
    user_queries: list[str] | None = None,
    existing_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Query facts from storage with validation.

    Args:
        storage: Storage backend instance
        embedder: Embeddings model for query encoding
        model: LLM model for query breaking
        credentials: Authentication credentials
        query_type: Type of query ('core' or 'context')
        filter_namespaces: Namespace filters for core facts
        match_count: Maximum number of facts to return
        user_queries: User queries for context-specific facts
        existing_ids: IDs to exclude from results

    Returns:
        List of fact dictionaries
    """
    # Validation
    if storage is None:
        logger.warning("Storage not initialized; cannot query facts")
        return []

    try:
        if query_type == "core":
            # Query core facts by namespace
            if filter_namespaces is None:
                logger.warning("No filter_namespaces provided for core facts query")
                return []

            facts = storage.backend.query_facts(
                credentials=credentials,
                filter_namespaces=filter_namespaces,
                match_count=match_count or 30,
            )
            logger.debug(f"Loaded {len(facts)} core facts")
            return facts

        elif query_type == "context":
            # Query context-specific facts using embeddings
            if embedder is None or model is None:
                logger.warning("Embedder or model not initialized; cannot query context facts")
                return []

            if not user_queries:
                logger.debug("No user queries provided for context facts")
                return []

            existing_ids = existing_ids or []
            all_facts = []

            # Break each query into atomic queries for better matching
            all_atomic_queries: list[str] = []
            for user_query in user_queries:
                atomic_queries = break_query_into_atomic(
                    model=model,
                    user_query=user_query,
                )
                all_atomic_queries.extend(atomic_queries)
                logger.debug(f"Query '{user_query[:50]}...' broke into {len(atomic_queries)} atomic queries")

            # Query facts using atomic queries
            for atomic_query in all_atomic_queries:
                try:
                    query_embedding = embedder.embed_query(atomic_query)
                    facts = storage.backend.query_facts(
                        credentials=credentials,
                        query_embedding=query_embedding,
                        match_count=match_count,
                    )

                    # Add facts that aren't already present
                    for fact in facts:
                        if fact.get("content") and fact.get("id") and fact["id"] not in existing_ids:
                            all_facts.append(fact)
                            existing_ids.append(fact["id"])
                except Exception as e:
                    logger.error(f"Error querying facts for atomic query '{atomic_query[:50]}...': {e}")
                    continue

            logger.debug(f"Loaded {len(all_facts)} context-specific facts")
            return all_facts

        else:
            logger.error(f"Invalid query_type: {query_type}")
            return []

    except Exception as e:
        logger.error(f"Error querying {query_type} facts: {e}")
        return []


class ContextEngineer(AgentMiddleware[AgentState, ContextT]):
    """Context Engineer enhanced context for agents through memory extraction and management.

    This middleware wraps model calls to provide context engineering capabilities:
    - Extracts key memories and insights from conversation messages
    - Stores memories in flexible backends (PostgreSQL, Supabase, Firebase, SQLite, ...)
    - Monitors token counts to trigger extraction at appropriate intervals
    - Prepares context for future model calls with relevant historical information
    - Returns operation traces under 'langmiddle:context:trace' for backend monitoring

    Implementation roadmap:
    - Phase 1: Memory extraction and storage vis supported backends
    - Phase 2 (Current): Context retrieval and injection into model requests
    - Phase 3: Dynamic context formatting based on relevance scoring
    - Phase 4: Multi-backend support (vector DB, custom storage adapters)
    - Phase 5: Advanced context optimization (token budgeting, semantic compression)

    Attributes:
        model: The LLM model for context analysis and memory extraction.
        embedder: Embedding model for memory representation.
        backend: Database backend to use. Currently only supports "supabase".
        extraction_prompt: System prompt guiding the facts extraction process.
        update_prompt: Custom prompt string guiding facts updating.
        core_prompt: Custom prompt string for core facts injection.
        memory_prompt: Custom prompt string for context-specific facts injection.
        max_tokens_before_summarization: Token threshold to trigger summarization.
        max_tokens_before_extraction: Token threshold to trigger extraction (None = use interval).
        extraction_interval: Extract facts every N agent completions (default: 3).
        summary_prompt: Prompt template for generating conversation summaries.
        token_counter: Function to count tokens in messages.
        embeddings_cache: Cache for reusing embeddings to improve performance.

    Note:
        Current implementation includes both memory extraction/storage (Phase 1)
        and context retrieval/injection (Phase 2). Future versions will add
        dynamic formatting and multi-backend support.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        embedder: str | Embeddings,
        backend: str = "supabase",
        *,
        extraction_prompt: str = DEFAULT_FACTS_EXTRACTOR,
        update_prompt: str = DEFAULT_FACTS_UPDATER,
        core_namespaces: list[list[str]] = ALWAYS_LOADED_NAMESPACES,
        core_prompt: str = DEFAULT_BASIC_INFO_INJECTOR,
        memory_prompt: str = DEFAULT_FACTS_INJECTOR,
        max_tokens_before_summarization: int | None = 5000,
        max_tokens_before_extraction: int | None = None,
        extraction_interval: int = 3,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        token_counter: TokenCounter = count_tokens_approximately,
        model_kwargs: dict[str, Any] | None = None,
        embedder_kwargs: dict[str, Any] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        # Configuration objects
        extraction_config: ExtractionConfig | None = None,
        summarization_config: SummarizationConfig | None = None,
        context_config: ContextConfig | None = None,
    ) -> None:
        """Initialize the context engineer.

        Args:
            model: LLM model for context analysis and memory extraction.
            embedder: Embedding model for memory representation.
            backend: Database backend to use. Currently only supports "supabase".
            extraction_prompt: Custom prompt string guiding facts extraction.
            update_prompt: Custom prompt string guiding facts updating.
            core_namespaces: List of namespaces to always load into context.
            core_prompt: Custom prompt string for core facts injection.
            memory_prompt: Custom prompt string for context-specific facts injection.
            max_tokens_before_summarization: Token threshold to trigger summarization.
                If None, summarization is disabled. Default: 8000 tokens.
            max_tokens_before_extraction: Token threshold to trigger extraction.
                If None, uses extraction_interval instead.
            extraction_interval: Extract facts every N agent completions.
                Default: 3 (extract every 3rd completion). Reduces LLM overhead.
            summary_prompt: Prompt template for generating summaries.
                Uses official LangGraph DEFAULT_SUMMARY_PROMPT by default.
            token_counter: Function to count tokens in messages.
            model_kwargs: Additional keyword arguments for model initialization.
            embedder_kwargs: Additional keyword arguments for embedder initialization.
            backend_kwargs: Additional keyword arguments for backend initialization.
            extraction_config: Optional ExtractionConfig object (overrides individual params).
            summarization_config: Optional SummarizationConfig object (overrides individual params).
            context_config: Optional ContextConfig object (overrides individual params).

        Note:
            Operations return trace logs under the 'langmiddle:context:trace' key
            for backend monitoring and debugging.

            Configuration objects provide a cleaner API but are optional.
            Individual parameters are maintained for backward compatibility.
        """
        super().__init__()

        # Build configuration objects from params or use provided
        self._extraction_config = extraction_config or ExtractionConfig(
            interval=extraction_interval,
            max_tokens=max_tokens_before_extraction,
            prompt=extraction_prompt,
            update_prompt=update_prompt,
        )

        self._summarization_config = summarization_config or SummarizationConfig(
            max_tokens=max_tokens_before_summarization or 5000,
            prompt=summary_prompt,
        )

        self._context_config = context_config or ContextConfig(
            core_namespaces=core_namespaces,
            core_prompt=core_prompt,
            memory_prompt=memory_prompt,
        )

        # Internal state management
        self._state = _MiddlewareState()

        # Public attributes for backward compatibility
        self.max_tokens_before_summarization: int | None = self._summarization_config.max_tokens
        self.max_tokens_before_extraction: int | None = self._extraction_config.max_tokens
        self.extraction_interval: int = self._extraction_config.interval
        self.summary_prompt: str = self._summarization_config.prompt
        self.extraction_prompt = self._extraction_config.prompt
        self.update_prompt = self._extraction_config.update_prompt
        self.memory_prompt = self._context_config.memory_prompt
        self.core_prompt = self._context_config.core_prompt
        self.core_namespaces = self._context_config.core_namespaces
        self.token_counter: TokenCounter = token_counter

        # Ensure valid backend and model configuration
        if backend.lower() != "supabase":
            logger.warning(f"Invalid backend: {backend}. Using default backend 'supabase'.")
            backend = "supabase"

        self.backend: str = backend.lower()

        self.model: BaseChatModel | None = None
        self.embedder: Embeddings | None = None
        self.storage: Any = None
        self.embeddings_cache: dict[str, list[float]] = self._state.embeddings_cache  # Backward compat

        # Initialize LLM model
        if isinstance(model, str):
            try:
                if model_kwargs is None:
                    model_kwargs = {}
                if "temperature" not in model_kwargs:
                    model_kwargs["temperature"] = 0.0  # Keep temperature low for consistent extractions
                model = init_chat_model(model, **model_kwargs)
            except Exception as e:
                logger.error(f"Error initializing chat model '{model}': {e}.")
                return

        if isinstance(model, BaseChatModel):
            self.model = model

        # Cache structured output models for prompt caching optimization
        # Reusing these models enables LLM provider caching (e.g., Anthropic prompt caching)
        self.extraction_model: Runnable | None = None
        self.actions_model: Runnable | None = None
        if self.model is not None:
            try:
                self.extraction_model = self.model.with_structured_output(ExtractedFacts)
                self.actions_model = self.model.with_structured_output(FactsActions)
                logger.debug("Cached structured output models for extraction and actions")
            except Exception as e:
                logger.warning(f"Failed to create structured output models: {e}")

        # Initialize embedding model
        if isinstance(embedder, str):
            try:
                if embedder_kwargs is None:
                    embedder_kwargs = {}
                embedder = init_embeddings(embedder, **embedder_kwargs)
            except Exception as e:
                logger.error(f"Error initializing embeddings model '{embedder}': {e}.")
                return

        if isinstance(embedder, Embeddings):
            self.embedder = embedder

        # Initialize storage backend
        if self.model is not None and self.embedder is not None:
            try:
                # For now, we don't pass credentials here - they'll be provided per-request
                self.storage = ChatStorage.create(backend, **(backend_kwargs or {}))
                logger.debug(f"Initialized storage backend: {backend}")
            except Exception as e:
                logger.error(f"Failed to initialize storage backend '{backend}': {e}")
                self.storage = None

        if self.model is None or self.embedder is None:
            logger.error(f"Initiation failed - the middleware {self.name} will be skipped during execution.")
        else:
            logger.info(
                f"Initialized middleware {self.name} with model {self.model.__class__.__name__} / "
                f"embedder: {self.embedder.__class__.__name__} / backend: {self.backend}."
            )

    # === Property Accessors ===

    @property
    def turn_count(self) -> int:
        """Number of agent turns processed."""
        return self._state.turn_count

    @property
    def extraction_count(self) -> int:
        """Number of fact extractions performed."""
        return self._state.extraction_count

    @property
    def extraction_config(self) -> ExtractionConfig:
        """Configuration for extraction behavior."""
        return self._extraction_config

    @property
    def summarization_config(self) -> SummarizationConfig:
        """Configuration for summarization behavior."""
        return self._summarization_config

    @property
    def context_config(self) -> ContextConfig:
        """Configuration for context injection behavior."""
        return self._context_config

    # === Cache Management ===

    def clear_embeddings_cache(self) -> None:
        """Clear the embeddings cache to free memory."""
        self._state.embeddings_cache.clear()
        logger.debug("Embeddings cache cleared")

    def reset_session_state(self) -> None:
        """Reset per-session state (turn count, current facts) while keeping caches."""
        self._state.reset_session_state()
        logger.debug("Session state reset")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the embeddings cache.

        Returns:
            Dictionary with cache statistics including size and sample keys
        """
        return {
            "size": len(self._state.embeddings_cache),
            "sample_keys": list(self._state.embeddings_cache.keys())[:5] if self._state.embeddings_cache else [],
        }

    # === Summarization Operations ===

    def _should_summarize(self, messages: list[AnyMessage]) -> bool:
        """Determine if summarization should be triggered.

        Args:
            messages: List of conversation messages.

        Returns:
            True if summarization should run, False otherwise.
        """
        if self._summarization_config.max_tokens is None:
            return False

        if not messages:
            return False

        # Skip if all messages have already been summarized
        if all(msg.id in self._state.summerized_msg_ids for msg in messages if msg.id is not None):
            return False

        total_tokens = self.token_counter(messages)

        return total_tokens >= self._summarization_config.max_tokens

    # === Extraction Operations ===

    def _should_extract(self, messages: list[AnyMessage]) -> bool:
        """Determine if extraction should be triggered based on multiple strategies.

        Implements smart extraction triggers:
        1. Interval-based: Extract every N turns (default: every 3 completions)
        2. Token-based: Extract when recent messages exceed threshold

        Args:
            messages: List of conversation messages.

        Returns:
            True if extraction should run, False otherwise.
        """
        if not messages:
            return False

        # Strategy 1: Interval-based extraction (default mode)
        # Extract every N completions to reduce overhead
        self._state.turn_count += 1
        if self._state.turn_count % self._extraction_config.interval == 0:
            logger.debug(f"Extraction triggered by interval (turn {self._state.turn_count})")
            return True

        # Strategy 2: Token-based extraction (fallback)
        # Extract if recent messages have significant content
        if self._extraction_config.max_tokens is not None:
            recent_tokens = self.token_counter(messages)
            if recent_tokens >= self._extraction_config.max_tokens:
                logger.debug(f"Extraction triggered by token threshold ({recent_tokens} tokens)")
                return True

        return False

    # === Context Operations ===

    def _summarize_conversation(
        self,
        messages: list[AnyMessage],
        prev_summary: str = "",
    ) -> str | None:
        """Generate a summary of conversation messages.

        Args:
            messages: list of messages to summarize.

        Returns:
            Summary text, or None if summarization fails.
        """
        if not messages or self.model is None:
            return None

        try:
            res = messages_summary(
                model=self.model,
                messages=messages,
                prev_summary=prev_summary,
                summary_prompt=self._summarization_config.prompt,
            )
            self._state.summerized_msg_ids.update(
                msg.id for msg in messages if msg.id is not None
            )
            return res
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None

    # === Fact Extraction Operations ===

    def _extract_facts(self, messages: list[AnyMessage]) -> list[dict] | None:
        """Extract facts from conversation messages.

        Args:
            messages: list of conversation messages.

        Returns:
            List of extracted facts as dictionaries, or None on failure.
        """
        if self.model is None:
            logger.error("Model not initialized for fact extraction.")
            return None

        # Use cached extraction model if available for prompt caching optimization
        model_to_use = self.extraction_model if self.extraction_model is not None else self.model

        extracted = extract_facts(
            model=model_to_use,
            extraction_prompt=self._extraction_config.prompt,
            messages=messages,
        )
        if extracted is None:
            logger.error("Fact extraction failed.")
            return None

        return [fact.model_dump() for fact in extracted.facts]

    # === Fact Query Operations ===

    def _query_existing_facts(
        self,
        new_facts: list[dict],
        user_id: str,
        credentials: dict[str, Any],
    ) -> list[dict]:
        """Query existing facts from storage using embeddings and namespace filtering.

        This is a wrapper around the standalone query_existing_facts function.

        Args:
            new_facts: List of newly extracted facts
            user_id: User identifier
            credentials: Credentials for storage backend

        Returns:
            List of existing relevant facts from storage
        """
        if self.storage is None or self.embedder is None:
            return []

        return query_existing_facts(
            storage_backend=self.storage.backend,
            credentials=credentials,
            embedder=self.embedder,
            new_facts=new_facts,
            user_id=user_id,
            embeddings_cache=self._state.embeddings_cache,
        )

    def _determine_actions(
        self,
        new_facts: list[dict],
        existing_facts: list[dict],
    ) -> list[dict] | None:
        """Determine what actions to take on facts (ADD, UPDATE, DELETE, NONE).

        Args:
            new_facts: List of newly extracted facts
            existing_facts: List of existing facts from storage

        Returns:
            List of actions to take, or None on failure
        """
        if self.model is None:
            logger.error("Model not initialized for action determination.")
            return None

        try:
            # Use cached actions model if available for prompt caching optimization
            model_to_use = self.actions_model if self.actions_model is not None else self.model

            actions = get_actions(
                model=model_to_use,
                update_prompt=self._extraction_config.update_prompt,
                current_facts=existing_facts,
                new_facts=new_facts,
            )

            if actions is None:
                logger.error("Failed to determine actions for facts")
                return None

            return [action.model_dump() for action in actions.actions]

        except Exception as e:
            logger.error(f"Error determining facts actions: {e}")
            return None

    def _apply_actions(
        self,
        actions: list[dict],
        user_id: str,
        credentials: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply fact actions to storage.

        This is a wrapper around the standalone apply_fact_actions function.

        Args:
            actions: List of action dictionaries from get_actions
            user_id: User identifier
            credentials: Credentials for storage backend

        Returns:
            Dictionary with action statistics and results
        """
        if self.storage is None or self.embedder is None:
            logger.error("Storage or embedder not initialized")
            return {
                "added": 0,
                "updated": 0,
                "deleted": 0,
                "skipped": 0,
                "errors": ["Storage not initialized"],
            }

        return apply_fact_actions(
            storage_backend=self.storage.backend,
            credentials=credentials,
            embedder=self.embedder,
            user_id=user_id,
            actions=actions,
            embeddings_cache=self._state.embeddings_cache,
            model=self.model,
        )

    # === Lifecycle Hooks ===

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Extract and manage facts after agent execution completes.

        This hook is called after each agent run, extracting facts from
        the conversation and managing them in the storage backend.

        Filters out summary-tagged messages to avoid extracting facts from
        compressed summaries, which would lose detail and accuracy.

        Args:
            state: Current agent state containing messages
            runtime: Runtime context with user_id and auth_token

        Returns:
            Dict with trace logs under 'langmiddle:context:trace' key, or None
        """
        # Get messages and check if extraction should run
        messages: list[AnyMessage] = state.get("messages", [])
        if not self._should_extract(messages):
            return None

        # Ensure storage is initialized
        if self.storage is None or self.model is None or self.embedder is None:
            logger.warning("Context engineer not fully initialized; skipping extraction")
            return None

        # Filter out summary messages - never extract facts from summaries
        extractable_messages = [
            msg for msg in messages
            if not msg.additional_kwargs.get(SUMMARY_TAG)
        ]

        if not extractable_messages:
            logger.debug("No extractable messages after filtering summaries")
            return None

        # Validate storage and authenticate
        user_id, credentials, error_msg = _validate_storage_and_auth(
            storage=self.storage,
            runtime=runtime,
            backend=self.backend,
        )
        if error_msg or user_id is None or credentials is None:
            logger.error(f"Validation failed: {error_msg}")
            return {LOGS_KEY: [f"ERROR: {error_msg}"]}

        trace_logs = []
        self._state.extraction_count += 1

        try:
            # Step 1: Extract facts from non-summary messages
            logger.debug(f"Extracting facts from {len(extractable_messages)} messages (filtered {len(messages) - len(extractable_messages)} summaries)")
            new_facts = self._extract_facts(extractable_messages)

            if not new_facts:
                logger.debug("No facts extracted from conversation")
                return None

            trace_logs.append(f"Extracted {len(new_facts)} new facts")
            logger.info(f"Extracted {len(new_facts)} facts")

            # Step 2: Query existing facts
            existing_facts = self._query_existing_facts(new_facts, user_id, credentials)
            if existing_facts:
                logger.debug(f"Found {len(existing_facts)} existing related facts")

            # Step 3: Determine actions
            actions = self._determine_actions(new_facts, existing_facts)

            if not actions:
                # If no actions determined, just insert new facts
                contents = [f["content"] for f in new_facts if f.get("content")]

                if not contents:
                    logger.warning("No valid content in new facts to insert")
                    return None

                try:
                    embeddings = self.embedder.embed_documents(contents)

                    # Validate embeddings
                    if not embeddings or not all(embeddings):
                        logger.error("Failed to generate embeddings for facts")
                        trace_logs.append("ERROR: Failed to generate embeddings")
                        return {LOGS_KEY: trace_logs}

                    # Ensure all embeddings have the same dimension
                    embedding_dims = [len(emb) for emb in embeddings if emb]
                    if not embedding_dims or len(set(embedding_dims)) > 1:
                        dims_info = set(embedding_dims) if embedding_dims else 'empty'
                        logger.error(f"Inconsistent embedding dimensions: {dims_info}")
                        trace_logs.append(f"ERROR: Inconsistent embedding dimensions: {dims_info}")
                        return {LOGS_KEY: trace_logs}

                    model_dimension = embedding_dims[0]

                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    trace_logs.append(f"ERROR: Failed to generate embeddings - {e}")
                    return {LOGS_KEY: trace_logs}

                result = self.storage.backend.insert_facts(
                    credentials=credentials,
                    user_id=user_id,
                    facts=new_facts,
                    embeddings=embeddings,
                    model_dimension=model_dimension,
                )

                inserted = result.get("inserted_count", 0)
                if inserted > 0:
                    trace_logs.append(f"Inserted {inserted} facts")
                    logger.info(f"Inserted {inserted} new facts")

                if result.get("errors"):
                    for error in result["errors"]:
                        trace_logs.append(f"ERROR: {error}")
                        logger.error(f"Fact insertion error: {error}")
            else:
                # Step 4: Apply actions
                stats = self._apply_actions(actions, user_id, credentials)

                # Log statistics for important operations
                total_changes = stats["added"] + stats["updated"] + stats["deleted"]
                if total_changes > 0:
                    summary = f"Facts: +{stats['added']} ~{stats['updated']} -{stats['deleted']}"
                    trace_logs.append(summary)
                    logger.info(summary)

                # Log errors
                for error in stats.get("errors", []):
                    trace_logs.append(f"ERROR: {error}")
                    logger.error(f"Fact management error: {error}")

        except Exception as e:
            error_msg = f"Unexpected error during fact extraction: {e}"
            trace_logs.append(f"ERROR: {error_msg}")
            logger.error(error_msg)

        return {LOGS_KEY: trace_logs} if trace_logs else None

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Context engineering before agent execution.

        Loads and injects relevant memories (core facts and context-specific facts)
        into the message history before the agent processes the request. Also handles
        conversation summarization when token limits are approached.

        Message structure after processing:
        1. SystemMessage [langmiddle/context] - Core + context facts (cached)
        2. HumanMessage [langmiddle/summary] - Summary of old messages (if needed)
        3. Recent messages (last 50% after summarization)

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            Dict with modified messages and optional trace logs, or None
        """
        # Read always loaded namespaces from storage
        messages: list[AnyMessage] = state.get("messages", [])
        if not messages:
            return None

        try:
            # Split messages by context, summary and regular messages
            split_msgs: dict[str, list[AnyMessage]] = split_messages(messages, by_tags=[CONTEXT_TAG, SUMMARY_TAG])
            conversation_msgs: list[AnyMessage] = split_msgs.get("default", [])
            context_msgs: list[AnyMessage] = split_msgs.get(CONTEXT_TAG, [])
            summary_msgs: list[AnyMessage] = split_msgs.get(SUMMARY_TAG, [])
            logger.debug(f"Split messages into {len(context_msgs)} context, {len(summary_msgs)} summary, and {len(conversation_msgs)} conversation messages")
        except Exception as e:
            logger.error(f"Error splitting messages into context, summary, and conversation messages: {e}")
            return None

        if not conversation_msgs:
            logger.debug("No conversation messages found")
            return None

        trace_logs = []
        is_summarized = False

        # =======================================================
        # Step 1. Summarization
        # =======================================================
        try:
            prev_summary: str = "\n\n".join([
                "\n".join(message_string_contents(msg)).lstrip(self.summarization_config.prefix)
                for msg in summary_msgs
            ]).strip()
            if self._should_summarize(conversation_msgs):
                summary_text: str | None = self._summarize_conversation(
                    messages=conversation_msgs,
                    prev_summary=prev_summary,
                )
                if summary_text:
                    summary_msgs = [HumanMessage(
                        content=f'{self.summarization_config.prefix}{summary_text}'.strip(),
                        additional_kwargs={SUMMARY_TAG: True},
                        id=summary_msgs[0].id if len(summary_msgs) > 0 else None,
                    )]
                    logger.info("Conversation summarized")
                    is_summarized = True
        except Exception as e:
            logger.error(f"Error during summarization: {e}")

        # =======================================================
        # Step 2. Load Facts (Semantic Memories)
        # =======================================================

        # Validate storage and authenticate
        user_id, credentials, error_msg = _validate_storage_and_auth(
            storage=self.storage,
            runtime=runtime,
            backend=self.backend,
        )

        if error_msg or user_id is None or credentials is None:
            logger.error(f"Validation failed: {error_msg}")
            trace_logs.append(f"ERROR: {error_msg}")

        else:
            try:
                # Load core memories (cached after first load)
                if not self._state.core_facts:
                    self._state.core_facts = _query_facts_with_validation(
                        storage=self.storage,
                        embedder=self.embedder,
                        model=self.model,
                        credentials=credentials,
                        query_type="core",
                        filter_namespaces=self._context_config.core_namespaces,
                        match_count=20,
                    )

                curr_ids = [fact["id"] for fact in self._state.core_facts + self._state.current_facts if fact.get("id")]

                # Load context-specific memories using atomic query breaking
                user_queries: list[str] = message_string_contents(messages[-1])
                context_facts = _query_facts_with_validation(
                    storage=self.storage,
                    embedder=self.embedder,
                    model=self.model,
                    credentials=credentials,
                    query_type="context",
                    user_queries=user_queries,
                    existing_ids=curr_ids,
                )
                self._state.current_facts.extend(context_facts)

                # Build context message with core + current facts
                context_parts = []

                if self._state.core_facts:
                    context_parts.append(self._context_config.core_prompt.format(basic_info=formatted_facts(self._state.core_facts)))

                if self._state.current_facts:
                    logger.debug(f"Applying {len(self._state.current_facts)} context-specific facts")
                    context_parts.append(self._context_config.memory_prompt.format(facts=formatted_facts(self._state.current_facts)))

                # Handle context message
                if context_parts:
                    context_msgs = [
                        SystemMessage(
                            content="\n\n".join(context_parts),
                            additional_kwargs={CONTEXT_TAG: True},
                            id=context_msgs[0].id if len(context_msgs) > 0 else None,
                        )
                    ]
                    trace_logs.append("Updated context message")

                # Log summary of operations
                total_core = len(self._state.core_facts)
                total_context = len(self._state.current_facts)
                if total_core > 0 or total_context > 0:
                    summary = f"Injected {total_core} core + {total_context} context facts"
                    trace_logs.append(summary)
                    logger.info(summary)

            except Exception as e:
                logger.error(f"Error during context injection: {e}")
                return None

        cutoff_idx = 0
        if is_summarized:
            cutoff_idx = len(conversation_msgs) // 2
            # Maek sure human message was kept close to half of the regular messages
            for idx in range(cutoff_idx, len(conversation_msgs)):
                if isinstance(conversation_msgs[idx], HumanMessage):
                    cutoff_idx = idx
                    break
            if cutoff_idx >= len(conversation_msgs) - 1:  # No human message found, look backwards
                for idx in range(cutoff_idx, 0, -1):
                    if isinstance(conversation_msgs[idx], HumanMessage):
                        cutoff_idx = idx + 1
                        break

        result = {"messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *context_msgs,
            *summary_msgs,
            *conversation_msgs[cutoff_idx:],
        ]}

        if trace_logs:
            result[LOGS_KEY] = trace_logs

        return result
