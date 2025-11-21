"""Supabase authentication client wrapper."""

import os
from typing import Any, Optional

from supabase import Client, create_client

from ..utils.logging import get_graph_logger

logger = get_graph_logger(__name__)


class AuthClient:
    """Manages authentication with shared Supabase backend."""

    # Shared backend configuration
    SHARED_PROJECT_URL = os.getenv(
        "LANGMIDDLE_PROJECT_URL", "https://frfdkegapvchcnczyesb.supabase.co"
    )
    SHARED_ANON_KEY = os.getenv(
        "LANGMIDDLE_ANON_KEY",
        # This is a public anon key - safe to embed
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZyZmRrZWdhcHZjaGNuY3p5ZXNiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMxNTE1MjcsImV4cCI6MjA3ODcyNzUyN30.8br0FE9Dbk0dmiPCz8Lqzi4rb0r9vO_P6VU5S8uOtNk",
    )

    def __init__(self) -> None:
        self.client: Client = create_client(self.SHARED_PROJECT_URL, self.SHARED_ANON_KEY)
        self._session: Optional[dict[str, Any]] = None

    def register(
        self,
        email: str,
        password: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Register new user account.

        Args:
            email: User's email address
            password: User's password
            metadata: Optional user metadata

        Returns:
            Dictionary with user_id, email, access_token, refresh_token

        Raises:
            Exception: If registration fails
        """
        try:
            response = self.client.auth.sign_up(
                {"email": email, "password": password, "options": {"data": metadata or {}}}
            )

            if response.user:
                self._session = response.session.__dict__ if response.session else {}
                return {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "access_token": response.session.access_token if response.session else "",
                    "refresh_token": response.session.refresh_token if response.session else "",
                }
            else:
                raise Exception("Registration failed - no user returned")

        except Exception as e:
            logger.error(f"Registration error: {e}")
            raise Exception(f"Registration error: {str(e)}")

    def login(self, email: str, password: str) -> dict[str, Any]:
        """Login with email/password.

        Args:
            email: User's email address
            password: User's password

        Returns:
            Dictionary with user_id, email, access_token, refresh_token

        Raises:
            Exception: If login fails
        """
        try:
            response = self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )

            if not response.user or not response.session:
                raise Exception("Login failed - no session returned")

            self._session = response.session.__dict__
            return {
                "user_id": response.user.id,
                "email": response.user.email,
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
            }

        except Exception as e:
            logger.error(f"Login error: {e}")
            raise Exception(f"Login error: {str(e)}")

    def logout(self) -> None:
        """Logout current session."""
        try:
            self.client.auth.sign_out()
        except Exception as e:
            logger.warning(f"Logout warning: {e}")
        finally:
            self._session = None

    def refresh_session(self, refresh_token: str) -> dict[str, Any]:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from previous session

        Returns:
            Dictionary with new access_token and refresh_token

        Raises:
            Exception: If token refresh fails
        """
        try:
            response = self.client.auth.refresh_session(refresh_token)
            self._session = response.session.__dict__ if response.session else {}

            return {
                "access_token": response.session.access_token if response.session else "",
                "refresh_token": response.session.refresh_token if response.session else "",
            }
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            raise Exception(f"Token refresh error: {str(e)}")

    def get_user(self) -> Optional[dict[str, Any]]:
        """Get current user info.

        Returns:
            Dictionary with user_id, email, metadata or None if not authenticated
        """
        try:
            user = self.client.auth.get_user()
            if user and user.user:
                return {
                    "user_id": user.user.id,
                    "email": user.user.email,
                    "metadata": user.user.user_metadata,
                }
        except Exception as e:
            logger.debug(f"Get user failed: {e}")
        return None

    def reset_password(self, email: str) -> None:
        """Send password reset email.

        Args:
            email: User's email address
        """
        try:
            self.client.auth.reset_password_email(email)
        except Exception as e:
            logger.error(f"Password reset error: {e}")
            raise Exception(f"Password reset error: {str(e)}")
