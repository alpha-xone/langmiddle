"""Session management for LangMiddle authentication."""

from typing import Any, Optional

from ..utils.logging import get_graph_logger
from .client import AuthClient
from .credentials import CredentialManager

logger = get_graph_logger(__name__)

# Global session state
_global_session: Optional[dict[str, Any]] = None
_auth_client: Optional[AuthClient] = None
_credential_manager: Optional[CredentialManager] = None


def _get_auth_client() -> AuthClient:
    """Get or create global auth client.

    Returns:
        AuthClient instance
    """
    global _auth_client
    if _auth_client is None:
        _auth_client = AuthClient()
    return _auth_client


def _get_credential_manager() -> CredentialManager:
    """Get or create global credential manager.

    Returns:
        CredentialManager instance
    """
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager


def register_user(
    email: str,
    password: str,
    save_credentials: bool = True,
) -> dict[str, Any]:
    """Register new user and optionally save credentials.

    Args:
        email: User's email address
        password: User's password
        save_credentials: Whether to save credentials to disk

    Returns:
        Dictionary with user_id, email, access_token, refresh_token

    Raises:
        Exception: If registration fails
    """
    global _global_session

    client = _get_auth_client()
    credentials = client.register(email, password)

    _global_session = credentials
    logger.info(f"User registered: {email}")

    if save_credentials:
        manager = _get_credential_manager()
        manager.save_credentials(credentials)

    return credentials


def login_user(
    email: str,
    password: str,
    save_credentials: bool = True,
) -> dict[str, Any]:
    """Login user and optionally save credentials.

    Args:
        email: User's email address
        password: User's password
        save_credentials: Whether to save credentials to disk

    Returns:
        Dictionary with user_id, email, access_token, refresh_token

    Raises:
        Exception: If login fails
    """
    global _global_session

    client = _get_auth_client()
    credentials = client.login(email, password)

    _global_session = credentials
    logger.info(f"User logged in: {email}")

    if save_credentials:
        manager = _get_credential_manager()
        manager.save_credentials(credentials)

    return credentials


def logout_user() -> None:
    """Logout current user and clear credentials."""
    global _global_session

    client = _get_auth_client()
    client.logout()

    manager = _get_credential_manager()
    manager.clear_credentials()

    _global_session = None
    logger.info("User logged out")


def get_current_user() -> Optional[dict[str, Any]]:
    """Get current authenticated user.

    Returns:
        Dictionary with user_id and email, or None if not authenticated
    """
    try:
        ensure_authenticated()
    except Exception:
        return None

    if _global_session:
        return {
            "user_id": _global_session["user_id"],
            "email": _global_session.get("email"),
        }

    client = _get_auth_client()
    return client.get_user()


def is_authenticated() -> bool:
    """Check if user is authenticated.

    Returns:
        True if user is authenticated
    """
    return _global_session is not None or _get_credential_manager().credentials_exist()


def ensure_authenticated() -> dict[str, Any]:
    """Ensure user is authenticated, load credentials if needed.

    Returns:
        Dictionary with current credentials

    Raises:
        Exception: If user is not authenticated
    """
    global _global_session

    # Already authenticated in memory
    if _global_session:
        return _global_session

    # Try loading saved credentials
    manager = _get_credential_manager()
    credentials = manager.load_credentials()

    if credentials:
        # Validate/refresh token
        client = _get_auth_client()
        try:
            refreshed = client.refresh_session(credentials["refresh_token"])
            credentials.update(refreshed)
            manager.save_credentials(credentials)
            _global_session = credentials
            logger.debug("Session refreshed from saved credentials")
            return credentials
        except Exception as e:
            # Token refresh failed, clear invalid credentials
            logger.warning(f"Token refresh failed: {e}")
            manager.clear_credentials()

    # Not authenticated
    raise Exception(
        "Not authenticated. Please run: langmiddle auth register\n"
        "Or use: from langmiddle import register_user; register_user(...)"
    )


def refresh_token() -> dict[str, Any]:
    """Manually refresh access token.

    Returns:
        Dictionary with refreshed credentials

    Raises:
        Exception: If token refresh fails
    """
    global _global_session

    credentials = ensure_authenticated()

    client = _get_auth_client()
    refreshed = client.refresh_session(credentials["refresh_token"])

    credentials.update(refreshed)

    manager = _get_credential_manager()
    manager.save_credentials(credentials)

    _global_session = credentials
    logger.info("Token refreshed")

    return credentials


def get_credentials_path() -> str:
    """Get path to credentials file.

    Returns:
        String path to credentials file
    """
    manager = _get_credential_manager()
    return str(manager.get_credentials_path())
