"""Configuration management for LangMiddle components.

This module provides centralized configuration classes to ensure consistency
across different middleware components (ChatSaver, ContextEngineer, etc.).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class StorageConfig:
    """Centralized configuration for storage backends.

    This config object can be shared across multiple middleware components
    to ensure they all use the same storage backend and settings.

    For zero-setup experience, use backend="langmiddle" (default).
    This automatically uses the LangMiddle platform (langmiddle.com).

    Attributes:
        backend: Storage backend to use ('langmiddle', 'sqlite', 'supabase', 'postgres', 'firebase').
        db_path: Path to SQLite database file (for 'sqlite' backend).
        enable_facts: Whether to enable semantic memory/facts tables.
        connection_string: PostgreSQL connection string (for 'postgres'/'supabase').
        supabase_url: Supabase project URL.
        supabase_key: Supabase API key.
        credentials_path: Path to Firebase credentials JSON.
        auto_create_tables: Whether to automatically create tables on startup.
    """

    backend: str = "langmiddle"
    db_path: str = ":memory:"
    enable_facts: bool = True
    connection_string: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    credentials_path: Optional[str] = None
    auto_create_tables: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert configuration to backend kwargs dictionary.

        Returns:
            Dictionary of arguments suitable for ChatStorage.create()
            or backend_kwargs parameters.
        """
        kwargs: dict[str, Any] = {
            "enable_facts": self.enable_facts,
            "auto_create_tables": self.auto_create_tables,
        }

        if self.backend == "sqlite":
            kwargs["db_path"] = self.db_path

        elif self.backend == "supabase":
            if self.supabase_url:
                kwargs["supabase_url"] = self.supabase_url
            if self.supabase_key:
                kwargs["supabase_key"] = self.supabase_key
            if self.connection_string:
                kwargs["connection_string"] = self.connection_string

        elif self.backend in ("postgres", "postgresql"):
            if self.connection_string:
                kwargs["connection_string"] = self.connection_string

        elif self.backend == "firebase":
            if self.credentials_path:
                kwargs["credentials_path"] = self.credentials_path

        return kwargs
