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

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from langchain.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from .memory.facts_manager import (
    ALWAYS_LOADED_NAMESPACES,
    apply_fact_actions,
    extract_facts,
    get_actions,
    query_existing_facts,
)
from .memory.facts_prompts import (
    DEFAULT_BASIC_INFO_INJECTOR,
    DEFAULT_FACTS_EXTRACTOR,
    DEFAULT_FACTS_INJECTOR,
    DEFAULT_FACTS_UPDATER,
)
from .storage import ChatStorage
from .utils.logging import get_graph_logger
from .utils.messages import split_messages
from .utils.runtime import auth_storage, get_user_id

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

load_dotenv()

logger = get_graph_logger(__name__)
# Disable propagation to avoid duplicate logs
logger._logger.propagate = False

CONTEXT_TAG = "langmiddle/context"
LOGS_KEY = "langmiddle:context:trace"


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
        max_tokens_before_extraction: Token threshold to trigger extraction (None = every completion).
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
        max_tokens_before_extraction: int | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
        model_kwargs: dict[str, Any] | None = None,
        embedder_kwargs: dict[str, Any] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
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
            max_tokens_before_extraction: Token threshold to trigger extraction.
                If None, extraction runs on every agent completion.
            token_counter: Function to count tokens in messages.
            model_kwargs: Additional keyword arguments for model initialization.
            embedder_kwargs: Additional keyword arguments for embedder initialization.
            backend_kwargs: Additional keyword arguments for backend initialization.

        Note:
            Operations return trace logs under the 'langmiddle:context:trace' key
            for backend monitoring and debugging.
        """
        super().__init__()

        self.max_tokens_before_extraction: int | None = max_tokens_before_extraction
        self.token_counter: TokenCounter = token_counter

        # Ensure valid backend and model configuration
        if backend.lower() != "supabase":
            logger.warning(f"Invalid backend: {backend}. Using default backend 'supabase'.")
            backend = "supabase"

        self.backend: str = backend.lower()
        self.user_id: str = ""

        self.extraction_prompt = extraction_prompt
        self.update_prompt = update_prompt
        self.memory_prompt = memory_prompt
        self.core_prompt = core_prompt
        self.core_namespaces = core_namespaces
        self.core_facts: list[dict[str, Any]] = []

        self.model: BaseChatModel | None = None
        self.embedder: Embeddings | None = None
        self.storage: Any = None
        self._extraction_count: int = 0
        self.embeddings_cache: dict[str, list[float]] = {}  # Cache for reusing embeddings

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

    def clear_embeddings_cache(self) -> None:
        """Clear the embeddings cache to free memory."""
        self.embeddings_cache.clear()
        logger.debug("Embeddings cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the embeddings cache.

        Returns:
            Dictionary with cache statistics including size and sample keys
        """
        return {
            "size": len(self.embeddings_cache),
            "sample_keys": list(self.embeddings_cache.keys())[:5] if self.embeddings_cache else [],
        }

    def _should_extract(self, messages: list[AnyMessage]) -> bool:
        """Determine if extraction should be triggered based on token count.

        Args:
            messages: List of conversation messages.

        Returns:
            True if extraction should run, False otherwise.
        """
        if not messages:
            return False

        if self.max_tokens_before_extraction is None:
            # Always extract if no threshold is set
            return True

        total_tokens: int = self.token_counter(messages)
        return total_tokens >= self.max_tokens_before_extraction

    def _extract_facts(self, messages: Sequence[AnyMessage | dict]) -> list[dict] | None:
        """Extract facts from conversation messages.

        Args:
            messages: Sequence of conversation messages.

        Returns:
            List of extracted facts as dictionaries, or None on failure.
        """
        if self.model is None:
            logger.error("Model not initialized for fact extraction.")
            return None

        extracted = extract_facts(
            model=self.model,
            extraction_prompt=self.extraction_prompt,
            messages=messages,
        )
        if extracted is None:
            logger.error("Fact extraction failed.")
            return None

        return [fact.model_dump() for fact in extracted.facts]

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
            embeddings_cache=self.embeddings_cache,
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
            actions = get_actions(
                model=self.model,
                update_prompt=self.update_prompt,
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
            embeddings_cache=self.embeddings_cache,
        )

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Extract and manage facts after agent execution completes.

        This hook is called after each agent run, extracting facts from
        the conversation and managing them in the storage backend.

        Args:
            state: Current agent state containing messages
            runtime: Runtime context with user_id and auth_token

        Returns:
            Dict with trace logs under 'langmiddle:context:trace' key, or None
        """
        # Check if we should extract
        messages: list[AnyMessage] = state.get("messages", [])
        if not self._should_extract(messages):
            return None

        # Ensure storage is initialized
        if self.storage is None or self.model is None or self.embedder is None:
            logger.warning("Context engineer not fully initialized; skipping extraction")
            return None

        # Extract context information
        user_id: str | None = get_user_id(
            runtime=runtime,
            backend=self.backend,
            storage_backend=self.storage.backend,
        )
        if not user_id:
            logger.error("Missing user_id in context; cannot extract facts")
            return {LOGS_KEY: ["ERROR: Missing user_id"]}

        # Prepare credentials and authenticate for storage backend
        auth_status: dict[str, Any] = auth_storage(
            runtime=runtime,
            backend=self.backend,
            storage_backend=self.storage.backend,
        )
        if "error" in auth_status:
            error_msg = auth_status["error"]
            logger.error(f"Authentication failed: {error_msg}")
            return {LOGS_KEY: [f"ERROR: Authentication failed - {error_msg}"]}

        credentials: dict[str, Any] = auth_status.get("credentials", {})
        trace_logs = []
        self._extraction_count += 1

        try:
            # Step 1: Extract facts from messages
            logger.debug(f"Extracting facts from {len(messages)} messages")
            new_facts = self._extract_facts(messages)

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
        into the message history before the agent processes the request.

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            Dict with modified messages and optional trace logs, or None
        """
        # Read always loaded namespaces from storage
        messages = state.get("messages", [])
        if not messages:
            return None

        if self.storage is None:
            return None

        # Extract context information
        user_id: str | None = get_user_id(
            runtime=runtime,
            backend=self.backend,
            storage_backend=self.storage.backend,
        )
        if not user_id:
            logger.error("Missing user_id in context; cannot load facts")
            return None

        # Prepare credentials and authenticate for storage backend
        auth_status: dict[str, Any] = auth_storage(
            runtime=runtime,
            backend=self.backend,
            storage_backend=self.storage.backend,
        )
        if "error" in auth_status:
            logger.error(f"Authentication failed: {auth_status['error']}")
            return None

        credentials: dict[str, Any] = auth_status.get("credentials", {})
        trace_logs = []

        try:
            # Split messages into context and recent messages
            context_messages, recent_messages = split_messages(messages, by_tag=CONTEXT_TAG)

            # Load core memories (cached after first load)
            if not self.core_facts:
                self.core_facts = self.storage.backend.query_facts(
                    credentials=credentials,
                    filter_namespaces=self.core_namespaces,
                    match_count=30,
                )
                if self.core_facts:
                    logger.debug(f"Loaded {len(self.core_facts)} core facts")

            core_ids = [fact["id"] for fact in self.core_facts]

            added_context = []
            if self.core_facts:
                formatted_core_facts = "\n".join(
                    f"[{' > '.join(fact['namespace'])}] {fact['content']}"
                    for fact in self.core_facts
                    if fact.get("content")
                )
                added_context.append(
                    SystemMessage(
                        content=self.core_prompt.format(basic_info=formatted_core_facts),
                        additional_kwargs={CONTEXT_TAG: True},
                    )
                )

            # Load context-specific memories
            current_facts = []
            if self.embedder is not None:
                queries: list[str] = []
                if isinstance(messages[-1].content, str):
                    queries.append(messages[-1].content)
                if isinstance(messages[-1].content, list):
                    for block in messages[-1].content:
                        if isinstance(block, str):
                            queries.append(block)

                for query in queries:
                    current_facts.extend([
                        fact for fact in self.storage.backend.query_facts(
                            credentials=credentials,
                            query_embedding=self.embedder.embed_query(query),
                        )
                        if fact.get("content") and fact["id"] not in core_ids
                    ])

                if current_facts:
                    logger.debug(f"Retrieved {len(current_facts)} context-specific facts")
                    formatted_cur_facts = "\n".join(
                        f"[{' > '.join(fact['namespace'])}] {fact['content']}"
                        if isinstance(fact.get("namespace"), list) and fact.get("namespace")
                        else fact['content']
                        for fact in current_facts
                    )
                    added_context.append(
                        SystemMessage(
                            content=self.memory_prompt.format(facts=formatted_cur_facts),
                            additional_kwargs={CONTEXT_TAG: True},
                        )
                    )

            if not added_context and context_messages:
                added_context.extend(context_messages)

            if not added_context:
                return None

            # Log summary of injected context
            total_core = len(self.core_facts)
            total_context = len(current_facts)
            if total_core > 0 or total_context > 0:
                summary = f"Injected {total_core} core + {total_context} context facts"
                trace_logs.append(summary)
                logger.info(summary)

            result = {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *added_context,
                    *recent_messages,
                ],
            }

            if trace_logs:
                result[LOGS_KEY] = trace_logs

            return result

        except Exception as e:
            logger.error(f"Error during context injection: {e}")
            return None
