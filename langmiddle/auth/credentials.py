"""Credential management for LangMiddle authentication."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..utils.logging import get_graph_logger

logger = get_graph_logger(__name__)


class CredentialManager:
    """Manages user credentials storage and retrieval."""

    def __init__(self, credentials_dir: Optional[Path] = None) -> None:
        """Initialize credential manager.

        Args:
            credentials_dir: Directory to store credentials.
                           Defaults to ~/.langmiddle
        """
        self.credentials_dir = credentials_dir or Path.home() / ".langmiddle"
        self.credentials_file = self.credentials_dir / "credentials.json"
        self.credentials_dir.mkdir(parents=True, exist_ok=True)

    def save_credentials(self, credentials: dict[str, Any]) -> None:
        """Save credentials to disk.

        Args:
            credentials: Dictionary containing user credentials
        """
        data = {
            "backend": "langmiddle",
            "user_id": credentials["user_id"],
            "email": credentials.get("email"),
            "access_token": credentials["access_token"],
            "refresh_token": credentials["refresh_token"],
            "project_url": "https://frfdkegapvchcnczyesb.supabase.co",
            "saved_at": datetime.utcnow().isoformat(),
        }

        with open(self.credentials_file, "w") as f:
            json.dump(data, f, indent=2)

        # Secure file permissions (Unix-like systems)
        if hasattr(os, "chmod"):
            try:
                os.chmod(self.credentials_file, 0o600)
                logger.debug("Set credentials file permissions to 0600")
            except Exception as e:
                logger.warning(f"Could not set file permissions: {e}")

        logger.info(f"Credentials saved to {self.credentials_file}")

    def load_credentials(self) -> Optional[dict[str, Any]]:
        """Load credentials from disk.

        Returns:
            Dictionary containing credentials or None if not found
        """
        if not self.credentials_file.exists():
            logger.debug("No credentials file found")
            return None

        try:
            with open(self.credentials_file, "r") as f:
                credentials = json.load(f)
            logger.debug("Credentials loaded successfully")
            return credentials
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return None

    def clear_credentials(self) -> None:
        """Remove saved credentials."""
        if self.credentials_file.exists():
            self.credentials_file.unlink()
            logger.info("Credentials cleared")
        else:
            logger.debug("No credentials to clear")

    def credentials_exist(self) -> bool:
        """Check if credentials are saved.

        Returns:
            True if credentials file exists
        """
        return self.credentials_file.exists()

    def get_credentials_path(self) -> Path:
        """Get path to credentials file.

        Returns:
            Path to credentials file
        """
        return self.credentials_file
