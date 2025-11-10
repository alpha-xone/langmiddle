from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from ..storage.base import ChatStorageBackend


def get_user_id(
    runtime: Runtime,
    backend: str | None = None,
    storage_backend: ChatStorageBackend | None = None,
) -> str | None:
    """Extract and validate user_id from runtime context.

    This function attempts to extract the user_id from the runtime context.
    If not found directly, it tries to extract it from the auth token based
    on the backend type.

    Args:
        runtime: LangGraph runtime context containing user_id and auth_token
        backend: Backend type ("supabase", "firebase", etc.) for token extraction
        storage_backend: Optional storage backend instance for user_id extraction

    Returns:
        User ID string if found, None otherwise

    Examples:
        >>> # Direct extraction from runtime
        >>> user_id = get_user_id(runtime)

        >>> # With backend fallback
        >>> user_id = get_user_id(runtime, backend="supabase")

        >>> # With storage backend instance
        >>> user_id = get_user_id(runtime, storage_backend=storage.backend)
    """
    # Try to get user_id directly from runtime context
    user_id: str | None = getattr(runtime.context, "user_id", None)

    # If not found and backend info is available, try to extract from auth token
    if not user_id and (backend or storage_backend):
        auth_token: str | None = getattr(runtime.context, "auth_token", None)

        # Use storage_backend if provided
        if storage_backend and auth_token:
            credentials: dict[str, Any] = {}
            if backend == "supabase" or (hasattr(storage_backend, "backend_type") and getattr(storage_backend, "backend_type") == "supabase"):
                credentials = {"jwt_token": auth_token}
            elif backend == "firebase" or (hasattr(storage_backend, "backend_type") and getattr(storage_backend, "backend_type") == "firebase"):
                credentials = {"id_token": auth_token}
            else:
                credentials = {"auth_token": auth_token}

            try:
                user_id = storage_backend.extract_user_id(credentials)
            except Exception:
                pass  # Extraction failed, user_id remains None

        # Fallback to backend string matching
        elif backend and auth_token:
            # Note: This requires the storage backend to be available
            # This is a simplified fallback that may not work without the backend instance
            pass

    return user_id


def auth_storage(
    runtime: Runtime,
    backend: str,
    storage_backend: ChatStorageBackend,
) -> dict[str, Any]:
    """Authenticate storage backend using runtime context.

    This function authenticates the provided storage backend using
    credentials extracted from the runtime context.

    Args:
        runtime: LangGraph runtime context containing auth_token
        backend: Backend type ("supabase", "firebase", etc.) for token extraction
        storage_backend: Storage backend instance for user_id extraction

    Returns:
        A dictionary indicating authentication status:
            - "authenticated": bool
            - "error": Exception (if any occurred during authentication)
            - "credentials": dict of used credentials

    Examples:
        >>> auth_status = authenticate_storage(runtime, backend, storage_backend)
    """
    credentials: dict[str, Any] = {}
    auth_token: str | None = getattr(runtime.context, "auth_token", None)

    if auth_token:
        credentials["auth_token"] = auth_token

    # Extract context information
    user_id: str | None = get_user_id(
        runtime=runtime,
        backend=backend,
        storage_backend=storage_backend,
    )
    auth_token: str | None = getattr(runtime.context, "auth_token", None)

    # Authenticate with storage backend
    credentials = {
        "user_id": user_id,
        "auth_token": auth_token,
    }
    try:
        credentials = storage_backend.prepare_credentials(**credentials)
        storage_backend.authenticate(credentials)
    except Exception as e:
        status = {"authenticated": False, "error": e, "credentials": {}}
        if user_id:
            status["credentials"]["user_id"] = user_id
        return status

    return {"authenticated": True, "credentials": credentials}
