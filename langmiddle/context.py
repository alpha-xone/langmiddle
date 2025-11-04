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

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from langmiddle.memory.facts_manager import (
    apply_fact_actions,
    extract_facts,
    get_actions,
    query_existing_facts,
)

from .memory.facts_prompts import DEFAULT_FACTS_EXTRACTOR, DEFAULT_FACTS_UPDATER
from .storage import ChatStorage
from .utils.logging import get_graph_logger

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

logger = get_graph_logger(__name__)
# Disable propagation to avoid duplicate logs
logger._logger.propagate = False


class ContextEngineer(AgentMiddleware[AgentState, ContextT]):
    """Context Engineer enhanced context for agents through memory extraction and management.

    This middleware wraps model calls to provide context engineering capabilities:
    - Extracts key memories and insights from conversation messages
    - Stores memories in flexible backends (PostgreSQL, Supabase, Firebase, SQLite, ...)
    - Monitors token counts to trigger extraction at appropriate intervals
    - Prepares context for future model calls with relevant historical information

    Implementation roadmap:
    - Phase 1 (Current): Memory extraction and storage vis supported backends
    - Phase 2: Context retrieval and injection into model requests
    - Phase 3: Dynamic context formatting based on relevance scoring
    - Phase 4: Multi-backend support (vector DB, custom storage adapters)
    - Phase 5: Advanced context optimization (token budgeting, semantic compression)

    Attributes:
        model: The LLM model for context analysis and memory extraction.
        embedder: Embedding model for memory representation.
        backend: Database backend to use. Currently only supports "supabase".
        extraction_prompt: System prompt guiding the facts extraction process.
        update_prompt: Custom prompt string guiding facts updating.
        max_tokens_before_extraction: Token threshold to trigger extraction (None = every completion).
        token_counter: Function to count tokens in messages.
        model_kwargs: Additional keyword arguments for model initialization.
        embedder_kwargs: Additional keyword arguments for embedder initialization.

    Note:
        Current implementation focuses on extraction and storage (Phase 1).
        Future versions will add context retrieval, dynamic formatting, and
        multi-backend support to complete the context engineering pipeline.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        embedder: str | Embeddings,
        backend: str = "supabase",
        *,
        extraction_prompt: str = DEFAULT_FACTS_EXTRACTOR,
        update_prompt: str = DEFAULT_FACTS_UPDATER,
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
            max_tokens_before_extraction: Token threshold to trigger extraction.
                If None, extraction runs on every agent completion.
            token_counter: Function to count tokens in messages.
            model_kwargs: Additional keyword arguments for model initialization.
            embedder_kwargs: Additional keyword arguments for embedder initialization.
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

        self.model: BaseChatModel | None = None
        self.embedder: Embeddings | None = None
        self.storage: Any = None
        self._extraction_count: int = 0
        self._logged_messages: set[str] = set()  # Track logged messages to avoid duplicates
        self.embeddings_cache: dict[str, list[float]] = {}  # Cache for reusing embeddings

        # Initialize LLM model
        if isinstance(model, str):
            try:
                if model_kwargs is None:
                    model_kwargs = {}
                if "temperature" not in model_kwargs:
                    model_kwargs["temperature"] = 0.1  # Keep temperature low for consistent extractions
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
        logger.info("Embeddings cache cleared")

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
            Dict with collected logs or None if no extraction occurred
        """
        # Check if we should extract
        messages: list[AnyMessage] = state.get("messages", [])

        if not self._should_extract(messages):
            return None

        # Ensure storage is initialized
        if self.storage is None or self.model is None or self.embedder is None:
            log_msg = "[after_agent] Context engineer not fully initialized; skipping extraction"
            if log_msg not in self._logged_messages:
                self._logged_messages.add(log_msg)
                return {"logs": [logger.warning(log_msg)]}
            return None

        # Extract context information
        user_id: str | None = getattr(runtime.context, "user_id", None)
        auth_token: str | None = getattr(runtime.context, "auth_token", None)

        if not user_id:
            if self.backend == "supabase":
                user_id = self.storage.backend.extract_user_id(
                    {"jwt_token": auth_token} if auth_token else {}
                )
            if self.backend == "firebase":
                user_id = self.storage.backend.extract_user_id(
                    {"id_token": auth_token} if auth_token else {}
                )

        if not user_id:
            log_msg = "[after_agent] Missing user_id in context; cannot extract facts"
            if log_msg not in self._logged_messages:
                self._logged_messages.add(log_msg)
                return {"logs": [logger.error(log_msg)]}
            return None

        # Authenticate with storage backend
        credentials = None
        try:
            credentials = self.storage.backend.prepare_credentials(
                user_id=user_id,
                auth_token=auth_token,
            )
            self.storage.backend.authenticate(credentials)
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            # Still set credentials even if authentication fails,
            # as some backends might not require explicit auth
            if credentials is None:
                credentials = {"user_id": user_id}

        graph_logs = []
        self._extraction_count += 1

        try:
            # Step 1: Extract facts from messages
            log_msg = f"[after_agent] Extracting facts from {len(messages)} messages"
            if log_msg not in self._logged_messages:
                graph_logs.append(logger.info(log_msg))
                self._logged_messages.add(log_msg)

            new_facts = self._extract_facts(messages)

            if not new_facts:
                log_msg = "[after_agent] No facts extracted from conversation"
                if log_msg not in self._logged_messages:
                    graph_logs.append(logger.debug(log_msg))
                    self._logged_messages.add(log_msg)
                return {"logs": graph_logs} if graph_logs else None

            log_msg = f"[after_agent] Extracted {len(new_facts)} facts"
            if log_msg not in self._logged_messages:
                graph_logs.append(logger.info(log_msg))
                self._logged_messages.add(log_msg)

            # Step 2: Query existing facts
            existing_facts = self._query_existing_facts(new_facts, user_id, credentials)

            if existing_facts:
                log_msg = f"[after_agent] Found {len(existing_facts)} existing related facts"
                if log_msg not in self._logged_messages:
                    graph_logs.append(logger.debug(log_msg))
                    self._logged_messages.add(log_msg)

            # Step 3: Determine actions
            actions = self._determine_actions(new_facts, existing_facts)

            if not actions:
                # If no actions determined, just insert new facts
                contents = [f["content"] for f in new_facts]
                embeddings = self.embedder.embed_documents(contents)
                model_dimension = len(embeddings[0]) if embeddings else 1536

                result = self.storage.backend.insert_facts(
                    credentials=credentials,
                    user_id=user_id,
                    facts=new_facts,
                    embeddings=embeddings,
                    model_dimension=model_dimension,
                )

                if result.get("inserted_count", 0) > 0:
                    log_msg = f"[after_agent] Inserted {result['inserted_count']} new facts"
                    graph_logs.append(logger.info(log_msg))

                if result.get("errors"):
                    for error in result["errors"]:
                        log_msg = f"[after_agent] Fact insertion error: {error}"
                        if log_msg not in self._logged_messages:
                            graph_logs.append(logger.error(log_msg))
                            self._logged_messages.add(log_msg)
            else:
                # Step 4: Apply actions
                stats = self._apply_actions(actions, user_id, credentials)

                # Log statistics
                if stats["added"] > 0 or stats["updated"] > 0 or stats["deleted"] > 0:
                    log_msg = (
                        f"[after_agent] Facts updated - "
                        f"Added: {stats['added']}, "
                        f"Updated: {stats['updated']}, "
                        f"Deleted: {stats['deleted']}"
                    )
                    graph_logs.append(logger.info(log_msg))

                # Log errors
                for error in stats.get("errors", []):
                    log_msg = f"[after_agent] Fact management error: {error}"
                    if log_msg not in self._logged_messages:
                        graph_logs.append(logger.error(log_msg))
                        self._logged_messages.add(log_msg)

        except Exception as e:
            log_msg = f"[after_agent] Unexpected error during fact extraction: {e}"
            if log_msg not in self._logged_messages:
                graph_logs.append(logger.error(log_msg))
                self._logged_messages.add(log_msg)

        return {"logs": graph_logs} if graph_logs else None
