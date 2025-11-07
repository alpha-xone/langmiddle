"""Context engineering middleware for LangChain agents.

This module provides middleware for engineering enhanced context by extracting and
managing conversation memories. It wraps model calls to enrich subsequent interactions
with relevant historical context, user preferences, and accumulated insights.

The context engineering process involves:
1. **Before Agent (Context Injection):**
   - Retrieve global context from ALWAYS_LOADED_NAMESPACES (user profile, preferences)
   - Query relevant facts based on recent conversation context
   - Generate conversation summary for long interactions
   - Restructure message queue with tagged context messages

2. **After Agent (Memory Extraction):**
   - Monitor conversation flow and token thresholds
   - Extract key memories and insights using LLM-based analysis
   - Store memories in flexible backends (PostgreSQL, Supabase, Firebase, SQLite)
   - Update/merge with existing facts to maintain consistency

This enables agents to maintain long-term memory and personalized understanding
across multiple conversation sessions with automatic context management.

Message Tagging:
- Messages are tagged using `additional_kwargs["langmiddle_tag"]` to enable replacement
- Tags: "langmiddle/context" (global), "langmiddle/facts" (relevant), "langmiddle/summary"
- Tagged messages are refreshed on each turn with up-to-date context
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from langmiddle.memory.facts_manager import (
    ALWAYS_LOADED_NAMESPACES,
    apply_fact_actions,
    extract_facts,
    get_actions,
    query_existing_facts,
)

from .context.builder import ContextComponents
from .context.models import ContextEngineerConfig, RetrievalContext
from .context.processors import ContextFormatter, FactDeduplicator, RelevanceFilter
from .context.queue_manager import MessageSeparator, QueueBuilder
from .context.retrievers import (
    ConversationSummarizer,
    GlobalContextRetriever,
    RelevantFactsRetriever,
)
from .memory.facts_prompts import DEFAULT_FACTS_EXTRACTOR, DEFAULT_FACTS_UPDATER
from .storage import ChatStorage
from .storage.base import ChatStorageBackend
from .utils.logging import get_graph_logger

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

logger = get_graph_logger(__name__)
# Disable propagation to avoid duplicate logs
logger._logger.propagate = False


class ContextEngineer(AgentMiddleware[AgentState, ContextT]):
    """Context Engineer for automatic memory extraction and context injection.

    This middleware provides comprehensive context management through two hooks:

    **before_agent (Context Injection):**
    - Automatically injects relevant context before each agent turn
    - Global context: User profile and preferences from ALWAYS_LOADED_NAMESPACES
    - Relevant facts: Context-aware retrieval based on recent conversation
    - Conversation summary: LLM-generated summary for long conversations
    - Message tagging: Uses additional_kwargs for efficient context updates

    **after_agent (Memory Extraction):**
    - Extracts key memories and insights from conversation messages
    - Stores memories in flexible backends (PostgreSQL, Supabase, Firebase, SQLite)
    - Monitors token counts to trigger extraction at appropriate intervals
    - Updates/merges with existing facts to maintain consistency

    Message Queue Structure:
        1. SystemMessage [langmiddle/context] ← Global user profile
        2. SystemMessage [langmiddle/facts] ← Relevant historical facts
        3. AIMessage [langmiddle/summary] ← Conversation summary (if needed)
        4. Recent messages (last N messages)
        5. Latest user input

    Attributes:
        model: LLM model for extraction, summarization, and fact management.
        embedder: Embedding model for vector similarity search.
        backend: Database backend (currently supports "supabase").
        extraction_prompt: System prompt guiding fact extraction.
        update_prompt: System prompt guiding fact updates.
        max_tokens_before_extraction: Token threshold to trigger extraction.
        max_recent_messages: Number of recent messages to keep in context.
        enable_context_injection: Whether to enable before_agent hook.
        token_counter: Function to count tokens in messages.
        embeddings_cache: Cache for reusing embeddings across queries.

    Example:
        ```python
        context_engineer = ContextEngineer(
            model="openai:gpt-4",
            embedder="openai:text-embedding-3-small",
            backend="supabase",
            enable_context_injection=True,  # Enable automatic context injection
            max_recent_messages=10,  # Keep last 10 messages
            max_tokens_before_extraction=2000,  # Extract after 2000 tokens
            backend_kwargs={
                "supabase_url": "...",
                "supabase_key": "...",
                "enable_facts": True,
            },
        )
        ```
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
        max_recent_messages: int = 10,
        enable_context_injection: bool = True,
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
            max_recent_messages: Maximum number of recent messages to keep in context.
            enable_context_injection: Whether to inject context before agent runs.
            token_counter: Function to count tokens in messages.
            model_kwargs: Additional keyword arguments for model initialization.
            embedder_kwargs: Additional keyword arguments for embedder initialization.
        """
        super().__init__()

        self.max_tokens_before_extraction: int | None = max_tokens_before_extraction
        self.max_recent_messages: int = max_recent_messages
        self.enable_context_injection: bool = enable_context_injection
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

        # Components for modular architecture (optional, used by builder pattern)
        self._components: ContextComponents | None = None

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

    @classmethod
    def _from_builder(
        cls,
        config: ContextEngineerConfig,
        model: BaseChatModel,
        embedder: Embeddings,
        storage: ChatStorageBackend,
        global_retriever: GlobalContextRetriever,
        relevant_retriever: RelevantFactsRetriever,
        summarizer: ConversationSummarizer,
        deduplicator: FactDeduplicator,
        relevance_filter: RelevanceFilter,
        formatter: ContextFormatter,
        queue_builder: QueueBuilder,
    ) -> "ContextEngineer[Any]":
        """Create ContextEngineer from builder (internal method).

        This is used by ContextEngineerBuilder to create instances
        with modular components.

        Args:
            config: Configuration object
            model: LLM model
            embedder: Embedding model
            storage: Storage backend
            global_retriever: Global context retriever
            relevant_retriever: Relevant facts retriever
            summarizer: Conversation summarizer
            deduplicator: Fact deduplicator
            relevance_filter: Relevance filter
            formatter: Context formatter
            queue_builder: Message queue builder

        Returns:
            Configured ContextEngineer instance
        """
        # Create instance using old-style __init__ for compatibility
        instance: ContextEngineer[Any] = cls.__new__(cls)
        super(ContextEngineer, instance).__init__()

        # Set basic attributes
        instance.max_tokens_before_extraction = config.max_tokens_before_extraction
        instance.max_recent_messages = config.max_recent_messages
        instance.enable_context_injection = config.enable_context_injection
        instance.token_counter = count_tokens_approximately
        instance.backend = "supabase"  # For now
        instance.user_id = ""
        instance.extraction_prompt = DEFAULT_FACTS_EXTRACTOR
        instance.update_prompt = DEFAULT_FACTS_UPDATER
        instance.model = model
        instance.embedder = embedder
        instance._extraction_count = 0
        instance._logged_messages = set()
        instance.embeddings_cache = {}

        # Create ChatStorage wrapper for compatibility
        instance.storage = type('Storage', (), {'backend': storage})()

        # Store components
        instance._components = ContextComponents(
            global_retriever=global_retriever,
            relevant_retriever=relevant_retriever,
            summarizer=summarizer,
            deduplicator=deduplicator,
            relevance_filter=relevance_filter,
            formatter=formatter,
            queue_builder=queue_builder,
        )

        logger.info(
            f"Initialized middleware {instance.name} with modular components / "
            f"model: {model.__class__.__name__} / embedder: {embedder.__class__.__name__}"
        )

        return instance

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

    def _extract_auth(
        self,
        runtime: Runtime[Any],
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Extract user_id and credentials from runtime context.

        Args:
            runtime: Runtime context

        Returns:
            Tuple of (user_id, credentials) or (None, None) if extraction failed
        """
        user_id: str | None = getattr(runtime.context, "user_id", None)
        auth_token: str | None = getattr(runtime.context, "auth_token", None)

        if not user_id:
            if self.backend == "supabase":
                user_id = self.storage.backend.extract_user_id(
                    {"jwt_token": auth_token} if auth_token else {}
                )
            elif self.backend == "firebase":
                user_id = self.storage.backend.extract_user_id(
                    {"id_token": auth_token} if auth_token else {}
                )

        if not user_id:
            log_msg = "Missing user_id in context"
            if log_msg not in self._logged_messages:
                self._logged_messages.add(log_msg)
                logger.error(log_msg)
            return None, None

        # Authenticate with storage backend
        credentials = None
        try:
            credentials = self.storage.backend.prepare_credentials(
                user_id=user_id,
                auth_token=auth_token,
            )
            self.storage.backend.authenticate(credentials)
        except Exception as e:
            logger.debug(f"Authentication note: {e}")
            # Still set credentials even if authentication fails
            if credentials is None:
                credentials = {"user_id": user_id}

        return user_id, credentials

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

    def _retrieve_global_context(
        self,
        user_id: str,
        credentials: dict[str, Any],
    ) -> list[dict]:
        """Retrieve global context facts from ALWAYS_LOADED_NAMESPACES.

        Args:
            user_id: User identifier
            credentials: Credentials for storage backend

        Returns:
            List of global context facts
        """
        if self.storage is None or self.embedder is None:
            return []

        try:
            global_facts = []
            for namespace in ALWAYS_LOADED_NAMESPACES:
                results = self.storage.backend.query_facts(
                    credentials=credentials,
                    query_embedding=None,  # No embedding, just namespace filter
                    user_id=user_id,
                    model_dimension=None,  # Query all dimensions
                    match_threshold=0.0,  # Get all facts in namespace
                    match_count=10,  # Limit per namespace
                    filter_namespaces=[namespace],
                )
                global_facts.extend(results)

            logger.debug(f"Retrieved {len(global_facts)} global context facts")
            return global_facts

        except Exception as e:
            logger.error(f"Error retrieving global context: {e}")
            return []

    def _retrieve_relevant_facts(
        self,
        messages: list[AnyMessage],
        user_id: str,
        credentials: dict[str, Any],
    ) -> list[dict]:
        """Retrieve facts relevant to recent conversation context.

        Args:
            messages: Recent conversation messages
            user_id: User identifier
            credentials: Credentials for storage backend

        Returns:
            List of relevant facts
        """
        if self.storage is None or self.embedder is None:
            return []

        try:
            # Get last few user/assistant messages for context
            recent_content = []
            for msg in messages[-5:]:  # Last 5 messages for relevance
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    recent_content.append(msg.content)

            if not recent_content:
                return []

            # Combine and embed recent context
            context_text = " ".join(recent_content)

            # Check cache first
            if context_text in self.embeddings_cache:
                context_embedding = self.embeddings_cache[context_text]
                logger.debug("Using cached embedding for context retrieval")
            else:
                context_embedding = self.embedder.embed_query(context_text)
                self.embeddings_cache[context_text] = context_embedding

            model_dimension = len(context_embedding)

            # Query for relevant facts
            relevant_facts = self.storage.backend.query_facts(
                credentials=credentials,
                query_embedding=context_embedding,
                user_id=user_id,
                model_dimension=model_dimension,
                match_threshold=0.70,  # Moderate threshold for relevance
                match_count=15,  # Get top relevant facts
                filter_namespaces=None,  # No namespace filter - search all
            )

            logger.debug(f"Retrieved {len(relevant_facts)} relevant facts")
            return relevant_facts

        except Exception as e:
            logger.error(f"Error retrieving relevant facts: {e}")
            return []

    def _deduplicate_facts(
        self,
        global_facts: list[dict],
        relevant_facts: list[dict],
    ) -> list[dict]:
        """Remove facts from relevant_facts that are duplicates of global_facts.

        Args:
            global_facts: Global context facts (always loaded)
            relevant_facts: Retrieved relevant facts

        Returns:
            Deduplicated list of relevant facts
        """
        if not global_facts or not relevant_facts:
            return relevant_facts

        # Create set of global fact IDs for fast lookup
        global_fact_ids = {fact.get("id") for fact in global_facts if fact.get("id")}

        # Also check content similarity for near-duplicates
        global_contents = {fact.get("content", "").lower().strip() for fact in global_facts}

        deduplicated = []
        for fact in relevant_facts:
            fact_id = fact.get("id")
            fact_content = fact.get("content", "").lower().strip()

            # Skip if same ID or very similar content
            if fact_id in global_fact_ids:
                logger.debug(f"Skipping duplicate fact by ID: {fact_id}")
                continue

            if fact_content in global_contents:
                logger.debug(f"Skipping duplicate fact by content: {fact_content[:50]}...")
                continue

            deduplicated.append(fact)

        logger.debug(
            f"Deduplicated {len(relevant_facts)} relevant facts to {len(deduplicated)} "
            f"(removed {len(relevant_facts) - len(deduplicated)} duplicates)"
        )
        return deduplicated

    def _filter_relevant_facts(
        self,
        facts: list[dict],
        recent_messages: list[AnyMessage],
    ) -> list[dict]:
        """Use LLM to filter facts for relevance to current conversation.

        This helps handle cases where embedding models may retrieve facts
        that aren't actually relevant to the current discussion.

        Args:
            facts: List of candidate facts
            recent_messages: Recent conversation messages for context

        Returns:
            Filtered list of relevant facts
        """
        if self.model is None or not facts or not recent_messages:
            return facts

        # If only a few facts, no need to filter
        if len(facts) <= 3:
            return facts

        try:
            # Extract recent conversation context
            recent_context = []
            for msg in recent_messages[-5:]:  # Last 5 messages
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                    recent_context.append(f"{role}: {msg.content}")

            conversation_context = "\n".join(recent_context)

            # Format facts for filtering
            facts_text = []
            for i, fact in enumerate(facts, 1):
                namespace = " > ".join(fact.get("namespace", []))
                content = fact.get("content", "")
                confidence = fact.get("confidence", 0.5)
                facts_text.append(f"{i}. [{namespace}] {content} (confidence: {confidence:.2f})")

            filter_prompt = f"""Given the recent conversation context, determine which of the following facts are RELEVANT to the current discussion.

A fact is relevant if it:
- Directly relates to topics being discussed
- Provides useful context for understanding the conversation
- Contains information that would help respond to the user

A fact is NOT relevant if it:
- Discusses completely unrelated topics
- Is only tangentially connected
- Would not add value to the current conversation

Recent Conversation:
{conversation_context}

Candidate Facts:
{chr(10).join(facts_text)}

Respond with ONLY the numbers of relevant facts, separated by commas (e.g., "1,3,5,7").
If none are relevant, respond with "NONE".
If all are relevant, respond with "ALL"."""

            response = self.model.invoke([SystemMessage(content=filter_prompt)])

            # Extract content
            if hasattr(response, "content"):
                filter_result = response.content
                if isinstance(filter_result, list):
                    filter_result = " ".join(str(item) for item in filter_result)
                elif not isinstance(filter_result, str):
                    filter_result = str(filter_result)
            else:
                filter_result = str(response)

            filter_result = filter_result.strip().upper()

            # Parse filtering result
            if filter_result == "NONE":
                logger.info("LLM filtered out all facts as not relevant")
                return []

            if filter_result == "ALL":
                logger.debug(f"LLM kept all {len(facts)} facts as relevant")
                return facts

            # Parse comma-separated numbers
            try:
                relevant_indices = [int(num.strip()) - 1 for num in filter_result.split(",")]
                filtered_facts = [facts[i] for i in relevant_indices if 0 <= i < len(facts)]

                logger.info(
                    f"LLM filtered {len(facts)} facts to {len(filtered_facts)} relevant facts"
                )
                return filtered_facts

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse LLM filter response: {filter_result}. Error: {e}")
                # On parse error, return all facts (safe default)
                return facts

        except Exception as e:
            logger.error(f"Error filtering facts with LLM: {e}")
            # On error, return all facts (safe default)
            return facts

    def _summarize_conversation(
        self,
        messages: list[AnyMessage],
    ) -> str | None:
        """Generate a summary of earlier conversation messages.

        Args:
            messages: Messages to summarize

        Returns:
            Summary text or None if no summary needed
        """
        if self.model is None or len(messages) <= self.max_recent_messages:
            return None

        try:
            # Messages to summarize (exclude recent ones)
            messages_to_summarize = messages[:-self.max_recent_messages]

            if len(messages_to_summarize) < 3:  # Too few to summarize
                return None

            summary_prompt = f"""Summarize the key points from this conversation history concisely.
Focus on: decisions made, information provided, user preferences expressed, and ongoing tasks.

Conversation to summarize:
{messages_to_summarize}

Provide a clear, structured summary (3-5 bullet points max):"""

            summary_response = self.model.invoke([SystemMessage(content=summary_prompt)])

            # Extract content properly
            if hasattr(summary_response, "content"):
                summary = summary_response.content
                # Handle list content (shouldn't happen, but be safe)
                if isinstance(summary, list):
                    summary = " ".join(str(item) for item in summary)
                elif not isinstance(summary, str):
                    summary = str(summary)
            else:
                summary = str(summary_response)

            logger.debug(f"Generated conversation summary ({len(messages_to_summarize)} messages)")
            return summary

        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return None

    def _format_facts_as_context(
        self,
        facts: list[dict],
        context_type: str = "general",
    ) -> str:
        """Format facts into readable context text.

        Args:
            facts: List of fact dictionaries
            context_type: Type of context (global, relevant)

        Returns:
            Formatted context string
        """
        if not facts:
            return ""

        lines = []
        if context_type == "global":
            lines.append("## User Profile & Preferences")
        else:
            lines.append("## Relevant Context from Previous Conversations")

        for fact in facts:
            namespace = fact.get("namespace", [])
            content = fact.get("content", "")
            confidence = fact.get("confidence", 0.5)

            # Format namespace as readable path
            ns_path = " > ".join(namespace) if namespace else "General"

            # Add fact with metadata
            lines.append(f"- [{ns_path}] {content} (confidence: {confidence:.2f})")

        return "\n".join(lines)

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Inject context before agent execution.

        This hook restructures the message queue:
        1. Backup and remove all messages
        2. Inject global context (system message with langmiddle/context tag)
        3. Inject retrieved facts (system message with langmiddle/facts tag)
        4. Inject conversation summary (AI message with langmiddle/summary tag)
        5. Add recent messages as-is

        On subsequent turns, tagged messages are replaced with fresh context.

        Args:
            state: Current agent state containing messages
            runtime: Runtime context with user_id and auth_token

        Returns:
            Dict with updated messages or None if no injection needed
        """
        if not self.enable_context_injection:
            return None

        messages: list[AnyMessage] = state.get("messages", [])

        if not messages:
            return None

        # Ensure storage is initialized
        if self.storage is None or self.embedder is None:
            log_msg = "[before_agent] Context engineer not fully initialized; skipping injection"
            if log_msg not in self._logged_messages:
                self._logged_messages.add(log_msg)
                logger.warning(log_msg)
            return None

        # Use modular components if available (builder pattern)
        if self._components is not None:
            return self._before_agent_modular(state, runtime)

        # Otherwise use legacy implementation
        return self._before_agent_legacy(state, runtime)

    def _before_agent_modular(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Context injection using modular components (new architecture).

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            Dict with updated messages or None
        """
        messages: list[AnyMessage] = state.get("messages", [])

        # Extract authentication
        user_id, credentials = self._extract_auth(runtime)
        if not user_id or not credentials:
            return None

        try:
            # Step 1: Separate tagged from regular messages
            separator = MessageSeparator()
            _, regular_messages = separator.separate(messages)

            # Step 2: Build retrieval context
            retrieval_ctx = RetrievalContext(
                user_id=user_id,
                credentials=credentials,
                messages=regular_messages,
                embeddings_cache=self.embeddings_cache,
            )

            # Step 3: Retrieve context
            assert self._components is not None  # Type narrowing
            global_facts = self._components.global_retriever.retrieve(retrieval_ctx)
            relevant_facts_raw = self._components.relevant_retriever.retrieve(retrieval_ctx)

            # Step 4: Process context
            relevant_facts_deduped = self._components.deduplicator.deduplicate(
                global_facts, relevant_facts_raw
            )
            relevant_facts = self._components.relevance_filter.filter(
                relevant_facts_deduped, regular_messages
            )

            # Step 5: Generate summary
            summary = self._components.summarizer.summarize(regular_messages)

            # Step 6: Format context
            global_text = self._components.formatter.format(global_facts, "global") if global_facts else None
            relevant_text = self._components.formatter.format(relevant_facts, "relevant") if relevant_facts else None

            # Step 7: Build queue
            new_messages = self._components.queue_builder.build(
                global_text,
                relevant_text,
                summary,
                regular_messages,
            )

            # Log statistics
            logger.info(
                f"[before_agent] Context injection (modular): "
                f"{len(global_facts)} global, "
                f"{len(relevant_facts)}/{len(relevant_facts_raw)} relevant, "
                f"{len(regular_messages)} messages"
            )

            return {"messages": new_messages}

        except Exception as e:
            logger.error(f"[before_agent] Error in modular context injection: {e}")
            return None

    def _before_agent_legacy(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Context injection using legacy implementation (backward compatibility).

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            Dict with updated messages or None
        """
        messages: list[AnyMessage] = state.get("messages", [])

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
            log_msg = "[before_agent] Missing user_id in context; cannot inject facts"
            if log_msg not in self._logged_messages:
                self._logged_messages.add(log_msg)
                logger.debug(log_msg)
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
            logger.error(f"[before_agent] Authentication failed: {e}")
            if credentials is None:
                credentials = {"user_id": user_id}

        try:
            # Step 1: Separate tagged messages from regular messages
            tagged_messages: dict[str, AnyMessage | None] = {
                "langmiddle/context": None,
                "langmiddle/facts": None,
                "langmiddle/summary": None,
            }
            regular_messages: list[AnyMessage] = []

            for msg in messages:
                additional_kwargs = getattr(msg, "additional_kwargs", {})
                msg_tag = additional_kwargs.get("langmiddle_tag")

                if msg_tag in tagged_messages:
                    # Track position but will replace with fresh content
                    tagged_messages[msg_tag] = msg
                else:
                    regular_messages.append(msg)

            # Step 2: Retrieve global context
            global_facts = self._retrieve_global_context(user_id, credentials)

            # Step 3: Retrieve relevant facts
            relevant_facts_raw = self._retrieve_relevant_facts(regular_messages, user_id, credentials)

            # Step 4: Deduplicate relevant facts (remove duplicates of global facts)
            relevant_facts_deduped = self._deduplicate_facts(global_facts, relevant_facts_raw)

            # Step 5: Filter relevant facts using LLM
            relevant_facts = self._filter_relevant_facts(relevant_facts_deduped, regular_messages)

            # Format context texts
            global_context_text = self._format_facts_as_context(global_facts, "global") if global_facts else ""
            relevant_context_text = self._format_facts_as_context(relevant_facts, "relevant") if relevant_facts else ""

            # Step 6: Generate conversation summary if needed
            summary_text = self._summarize_conversation(regular_messages)

            # Step 7: Build new message queue
            new_messages = []

            # Add global context (only if has facts)
            if global_facts and global_context_text:
                global_msg = SystemMessage(
                    content=global_context_text,
                    additional_kwargs={"langmiddle_tag": "langmiddle/context"}
                )
                new_messages.append(global_msg)
                logger.debug(f"[before_agent] Injected {len(global_facts)} global context facts")

            # Add relevant facts (only if has facts)
            if relevant_facts and relevant_context_text:
                facts_msg = SystemMessage(
                    content=relevant_context_text,
                    additional_kwargs={"langmiddle_tag": "langmiddle/facts"}
                )
                new_messages.append(facts_msg)
                logger.debug(f"[before_agent] Injected {len(relevant_facts)} relevant facts")

            # Add conversation summary
            if summary_text:
                summary_msg = AIMessage(
                    content=f"## Previous Conversation Summary\n{summary_text}",
                    additional_kwargs={"langmiddle_tag": "langmiddle/summary"}
                )
                new_messages.append(summary_msg)
                logger.debug("[before_agent] Injected conversation summary")

            # Add recent messages (last N messages)
            recent_messages = regular_messages[-self.max_recent_messages:]
            new_messages.extend(recent_messages)

            # Log statistics
            if relevant_facts_raw:
                logger.info(
                    f"[before_agent] Context injection complete: "
                    f"{len(global_facts)} global facts, "
                    f"{len(relevant_facts)} relevant facts "
                    f"(from {len(relevant_facts_raw)} retrieved, "
                    f"{len(relevant_facts_deduped)} after deduplication), "
                    f"{len(recent_messages)} recent messages"
                )
            else:
                logger.info(
                    f"[before_agent] Context injection complete: "
                    f"{len(global_facts)} global facts, "
                    f"0 relevant facts, "
                    f"{len(recent_messages)} recent messages"
                )

            return {"messages": new_messages}

        except Exception as e:
            logger.error(f"[before_agent] Error during context injection: {e}")
            return None

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
