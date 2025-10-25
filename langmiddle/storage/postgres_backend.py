"""
PostgreSQL storage backend implementation.

This module provides direct PostgreSQL implementation of the chat storage interface
using psycopg2 for database connections.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging import get_graph_logger
from .postgres_base import PostgreSQLBaseBackend

logger = get_graph_logger(__name__)

__all__ = ["PostgreSQLStorageBackend"]


class PostgreSQLStorageBackend(PostgreSQLBaseBackend):
    """Direct PostgreSQL implementation of chat storage backend."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        auto_create_tables: bool = False,
        load_from_env: bool = True,
    ):
        """
        Initialize PostgreSQL storage backend.

        Args:
            connection_string: PostgreSQL connection string (optional if using .env)
            auto_create_tables: Whether to automatically create tables if they don't exist (default: False)
            load_from_env: Whether to load connection string from .env file (default: True)

        Raises:
            ImportError: If psycopg2 dependencies are not installed
            ValueError: If connection string is not provided and not found in environment

        Example:
            # Using connection string directly
            storage = ChatStorage.create(
                "postgres",
                connection_string="postgresql://user:password@localhost:5432/dbname",
                auto_create_tables=True
            )

            # Using environment variables
            storage = ChatStorage.create("postgres")  # Loads from .env
        """
        # Try to import psycopg2
        try:
            from psycopg2 import pool  # noqa: F401
        except ImportError:
            raise ImportError(
                "psycopg2 dependencies not installed. "
                "Install with: pip install langmiddle[postgres] or pip install psycopg2-binary"
            )

        # Load from environment if requested
        if load_from_env and not connection_string:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                logger.debug("python-dotenv not installed, skipping .env file loading")

            connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

        # Validate connection string
        if not connection_string:
            raise ValueError(
                "PostgreSQL connection string not provided. Either:\n"
                "1. Pass connection_string parameter, or\n"
                "2. Set POSTGRES_CONNECTION_STRING environment variable, or\n"
                "3. Add it to a .env file in your project root\n\n"
                "Example: postgresql://user:password@localhost:5432/dbname"
            )

        self.connection_string = connection_string
        self._connection_pool = None

        # Initialize connection pool
        try:
            self._connection_pool = pool.SimpleConnectionPool(
                1, 10, connection_string  # min connections  # max connections
            )
            logger.debug("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise

        # Create tables if requested
        if auto_create_tables:
            sql_dir = Path(__file__).parent / "postgres"
            self._create_tables_with_psycopg2(connection_string, sql_dir)

    def _get_connection(self):
        """Get a connection from the pool."""
        if not self._connection_pool:
            raise RuntimeError("Connection pool not initialized")
        return self._connection_pool.getconn()

    def _return_connection(self, conn):
        """Return a connection to the pool."""
        if self._connection_pool:
            self._connection_pool.putconn(conn)

    def _execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
    ) -> Optional[Any]:
        """
        Execute a SQL query using psycopg2.

        Args:
            query: SQL query string
            params: Query parameters tuple
            fetch_one: Whether to fetch one result
            fetch_all: Whether to fetch all results

        Returns:
            Query results if fetch_one or fetch_all, None otherwise
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(query, params)

            if fetch_one:
                result = cursor.fetchone()
                cursor.close()
                conn.commit()
                return result
            elif fetch_all:
                results = cursor.fetchall()
                cursor.close()
                conn.commit()
                return results
            else:
                cursor.close()
                conn.commit()
                return None

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database query error: {e}")
            raise
        finally:
            if conn:
                self._return_connection(conn)

    def authenticate(self, credentials: Optional[Dict[str, Any]]) -> bool:
        """
        Authenticate with PostgreSQL.

        For direct PostgreSQL, authentication is handled at connection time.
        This method is a no-op that always returns True.

        Args:
            credentials: Not used for PostgreSQL (authentication via connection string)

        Returns:
            True (authentication handled by connection string)
        """
        logger.debug("PostgreSQL authentication handled by connection string")
        return True

    def extract_user_id(self, credentials: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Extract user ID from credentials.

        For direct PostgreSQL without external auth system, user_id must be
        provided directly in credentials.

        Args:
            credentials: Dict containing 'user_id' key

        Returns:
            User ID if found, None otherwise
        """
        if not credentials:
            return None

        user_id = credentials.get("user_id")
        if user_id:
            return user_id

        logger.debug("No user_id found in credentials")
        return None

    def close(self):
        """Close the connection pool and clean up resources."""
        if self._connection_pool:
            self._connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
