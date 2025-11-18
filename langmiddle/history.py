"""Chat history middleware for saving conversations to various storage backends.

This module provides LangChain v1 middleware for automatically persisting
chat messages to databases (SQLite, Supabase, Firebase) after each model response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import RemoveMessage
from langchain_core.messages import AnyMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from .storage import ChatStorage
from .utils.logging import get_graph_logger
from .utils.messages import filter_tool_messages
from .utils.runtime import get_user_id

load_dotenv()

logger = get_graph_logger(__name__)
# Disable propagation to avoid duplicate logs (LangGraph handles the logging)
logger._logger.propagate = False

LOGS_KEY = "langmiddle:history:trace"

__all__ = ["StorageContext", "ToolRemover", "ChatSaver"]


@dataclass
class StorageContext:
    """Context schema for chat storage middleware.

    This schema works across all storage backends (SQLite, Supabase, Firebase).

    Attributes:
        thread_id: Conversation thread identifier (required for all backends).
        user_id: User identifier (optional, used for multi-tenant scenarios).
        auth_token: Authentication token (optional, required only for Supabase/Firebase).
            - For Supabase: JWT token
            - For Firebase: ID token
            - For SQLite: Not used (pass None or empty string)

    Examples:
        SQLite (in-memory or file) - no token needed:

        ```python
        context = StorageContext(thread_id="thread-456", user_id="user-123")
        ```

        Supabase - requires JWT token:

        ```python
        context = StorageContext(
            thread_id="thread-456",
            user_id="user-123",
            auth_token="eyJ..."
        )
        ```

        Firebase - requires ID token:

        ```python
        context = StorageContext(
            thread_id="thread-456",
            user_id="user-123",
            auth_token="firebase_id_token..."
        )
        ```
    """

    thread_id: str
    user_id: str | None = None
    auth_token: str | None = None


class ToolRemover(AgentMiddleware[AgentState, ContextT]):
    """
    Middleware to remove tool messages from chat history.

    This middleware removes tool-related messages that shouldn't be saved:
    1. Messages with type 'tool'
    2. AI messages that trigger tool calls (finish_reason == 'tool_calls')

    Usage:
        agent = create_agent(
            model="openai:gpt-4o",
            tools=[...],
            middleware=[
                ToolRemover(),    # Remove before and after agent (default)
                ChatSaver(),      # Then save
            ],
            context_schema=ContextSchema,
        )
    """

    def __init__(self, when: str = "both"):
        """
        Initialize the tool message remover middleware.

        Args:
            when: When to filter messages - 'before', 'after', or 'both' (default: 'both')
        """
        super().__init__()
        if when not in ("before", "after", "both"):
            raise ValueError(
                f"Invalid 'when' value: {when}. Must be 'before', 'after', or 'both'"
            )
        self.when: str = when

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """
        Filter tool messages from the state before agent call.

        Args:
            state: Current agent state containing messages
            runtime: Runtime context

        Returns:
            Updated state dict with filtered messages
        """
        if self.when not in ("before", "both"):
            return None

        messages: list[AnyMessage] = state.get("messages", [])
        new_messages: list[AnyMessage] = filter_tool_messages(messages)

        # Only return update if we have messages to remove
        cnt_diff: int = len(messages) - len(new_messages)
        if cnt_diff > 0:
            logger.debug(f"[before_agent] Marked {cnt_diff} tool-related messages for removal")
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *new_messages,
                ]
            }

        return None

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """
        Filter tool messages from the state after agent call.

        Args:
            state: Current agent state containing messages
            runtime: Runtime context

        Returns:
            Updated state dict with filtered messages
        """
        if self.when not in ("after", "both"):
            return None

        messages: list[AnyMessage] = state.get("messages", [])
        new_messages: list[AnyMessage] = filter_tool_messages(messages)

        # Only return update if we have messages to remove
        cnt_diff: int = len(messages) - len(new_messages)
        if cnt_diff > 0:
            logger.debug(f"[after_agent] Marked {cnt_diff} tool-related messages for removal")
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *new_messages,
                ]
            }

        return None


class ChatSaver(AgentMiddleware[AgentState, ContextT]):
    """Middleware to save chat history to various storage backends after each model response.

    This middleware automatically captures and persists conversation history
    to the database, including message content, and metadata.
    Supports multiple storage backends: SQLite (default), Supabase, and Firebase.
    Returns operation traces under 'langmiddle:history:trace' for backend monitoring.

    Usage:
        Using SQLite in-memory (default - easiest to get started):

        ```python
        agent = create_agent(
            model="openai:gpt-4o",
            tools=[...],
            middleware=[ChatSaver()],
            context_schema=StorageContext,
        )
        # Invoke with context (auth_token optional for SQLite)
        agent.invoke(
            {"messages": [...]},
            context=StorageContext(user_id="user-123", thread_id="thread-123")
        )
        ```

        Using SQLite with file storage:

        ```python
        agent = create_agent(
            model="openai:gpt-4o",
            tools=[...],
            middleware=[ChatSaver(db_path="./chat.db")],
            context_schema=StorageContext,
        )
        ```

        Using Supabase (requires auth_token):

        ```python
        agent = create_agent(
            model="openai:gpt-4o",
            tools=[...],
            middleware=[ChatSaver(
                backend="supabase",
                supabase_url="https://your-project.supabase.co",
                supabase_key="your-anon-key",
            )],
            context_schema=StorageContext
        )
        # Invoke with JWT token for Supabase
        agent.invoke(
            {"messages": [...]},
            context=StorageContext(
                user_id="user-123",
                thread_id="thread-123",
                auth_token="eyJ..."
            )
        )
        ```

        Using Firebase (requires auth_token as ID token):

        ```python
        agent = create_agent(
            model="openai:gpt-4o",
            tools=[...],
            middleware=[ChatSaver(
                backend="firebase",
                credentials_path="/path/to/firebase-creds.json"
            )],
            context_schema=StorageContext,
        )
        # Invoke with ID token for Firebase
        agent.invoke(
            {"messages": [...]},
            context=StorageContext(
                user_id="user-123",
                thread_id="thread-123",
                auth_token="firebase_id_token..."
            )
        )
        ```
    """

    def __init__(
        self,
        save_interval: int = 1,
        backend: str = "sqlite",
        **backend_kwargs: Any,
    ) -> None:
        """Initialize chat history middleware.

        Args:
            save_interval: Save to database after every N model responses (default: 1).
                Must be >= 1.
            backend: Storage backend to use. Supported values: 'sqlite', 'supabase', 'firebase'.
                Defaults to 'sqlite'.
            **backend_kwargs: Backend-specific initialization parameters:
                - For SQLite: db_path (str, default: ":memory:" for in-memory database)
                - For Supabase: supabase_url (str), supabase_key (str), or client (optional)
                - For Firebase: credentials_path (str, optional)

        Raises:
            ValueError: If save_interval < 1 or backend is not supported.
            Exception: If storage backend initialization fails.

        Note:
            Save operations automatically track and skip duplicate messages using persistent
            message ID tracking. Trace logs are returned under 'langmiddle:history:trace'
            for backend monitoring and debugging.
        """
        super().__init__()

        if save_interval < 1:
            msg = f"save_interval must be >= 1, got {save_interval}"
            raise ValueError(msg)

        self.save_interval: int = save_interval
        self._model_call_count: int = 0
        self._saved_msg_ids: set[str] = set()  # Persistent tracking of saved message IDs

        # Set default db_path for SQLite if not provided
        if backend == "sqlite" and "db_path" not in backend_kwargs:
            backend_kwargs["db_path"] = ":memory:"

        # Initialize storage backend
        try:
            self.storage = ChatStorage.create(backend, **backend_kwargs)
            logger.info(f"Initialized middleware {self.name} with backend: {backend}")
        except Exception as e:
            logger.error(f"Failed to initialize storage backend '{backend}': {e}")
            logger.error(f"Initiation failed - the middleware {self.name} will be skipped during execution.")

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Save chat history after agent execution completes.

        This hook is called after each agent run completes, allowing us to
        persist the conversation state to the configured storage backend.

        Args:
            state: Current agent state containing messages.
            runtime: Runtime context with user_id, thread_id, and auth_token.

        Returns:
            Dict with trace logs under 'langmiddle:history:trace' key:
            {"langmiddle:history:trace": ["Saved 3 messages to thread-123 (skipped 2 duplicates)"]}
            Returns None if no save operation was performed.
        """
        # Increment call count
        self._model_call_count += 1

        # Only save on the configured interval
        if self._model_call_count % self.save_interval != 0:
            return None

        # Extract context information from runtime
        user_id: str | None = get_user_id(
            runtime=runtime,
            storage_backend=self.storage.backend if self.storage else None,
        )
        thread_id: str | None = getattr(runtime.context, "thread_id", None)
        auth_token: str | None = getattr(runtime.context, "auth_token", None)

        # Validate required context
        if not thread_id:
            logger.error("Missing thread_id in context; cannot save chat history")
            return {LOGS_KEY: ["ERROR: Missing thread_id"]}

        # Get messages from state
        messages: list[AnyMessage] = state.get("messages", [])

        if not messages:
            logger.debug(f"No messages to save for thread {thread_id}")
            return None

        custom_state = None
        for key, value in state.items():
            if key not in ("messages", "jump_to"):
                if custom_state is None:
                    custom_state = {}
                custom_state[key] = value

        # Prepare credentials based on available context
        credentials: dict[str, Any] = self._prepare_credentials(user_id, auth_token)

        # Save chat history using the storage backend
        result = self.storage.save_chat_history(
            thread_id=thread_id,
            credentials=credentials,
            messages=messages,
            user_id=user_id,
            saved_msg_ids=self._saved_msg_ids,  # Pass persistent set
            custom_state=custom_state,
        )

        # Update the persistent set with newly saved message IDs
        if "saved_msg_ids" in result:
            self._saved_msg_ids.update(result["saved_msg_ids"])

        # Log the result and collect trace logs
        return self._log_save_result(result, thread_id)

    def _prepare_credentials(
        self,
        user_id: str | None,
        auth_token: str | None,
    ) -> dict[str, Any]:
        """Prepare credentials dict based on available context and backend type.

        Args:
            user_id: User identifier.
            auth_token: Authentication token (JWT or ID token).

        Returns:
            Credentials dictionary with appropriate keys for the backend.
        """
        credentials: dict[str, Any] = {"user_id": user_id}

        if auth_token:
            # Add token with appropriate key based on backend type
            backend_type: str = type(self.storage.backend).__name__.lower()
            if "firebase" in backend_type:
                credentials["id_token"] = auth_token
            else:  # Supabase or other JWT-based backends
                credentials["jwt_token"] = auth_token

        return credentials

    def _log_save_result(
        self,
        result: dict[str, Any],
        thread_id: str,
    ) -> dict[str, Any] | None:
        """Log the save result and return trace logs.

        Args:
            result: Result dictionary from save_chat_history.
            thread_id: Thread identifier for logging.

        Returns:
            Dict with trace logs under 'langmiddle:history:trace' key, or None
        """
        trace_logs = []

        if result["success"]:
            if result["saved_count"] > 0:
                saved = result["saved_count"]
                skipped = result.get("skipped_count", 0)
                summary = f"Saved {saved} messages to thread {thread_id}"
                if skipped > 0:
                    summary += f" (skipped {skipped} duplicates)"
                trace_logs.append(summary)
                logger.info(summary)
            else:
                logger.debug(f"No new messages to save for thread {thread_id}")
        else:
            # Log errors
            for error in result["errors"]:
                trace_logs.append(f"ERROR: {error}")
                logger.error(f"Chat history save error: {error}")

        return {LOGS_KEY: trace_logs} if trace_logs else None
