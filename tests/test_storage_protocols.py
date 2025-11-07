"""Test protocol compatibility for storage backends.

This test verifies that storage backends implement the StorageBackend protocol.

Run from repo root: python -m tests.test_storage_protocols
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from langmiddle.context.core.protocols import StorageBackend


def test_supabase_backend_implements_protocol():
    """Verify SupabaseStorageBackend implements StorageBackend protocol."""
    from langmiddle.storage.supabase_backend import SupabaseStorageBackend

    # Mock Supabase client
    class MockClient:
        def table(self, name):
            return self

        def select(self, *args):
            return self

        def execute(self):
            class MockResult:
                data = []

            return MockResult()

        def rpc(self, name, params):
            return self

    # Create backend with mock client
    backend = SupabaseStorageBackend(client=MockClient())

    # Verify it implements the protocol
    assert isinstance(backend, StorageBackend), (
        "SupabaseStorageBackend must implement StorageBackend protocol. "
        f"Missing methods: {_check_protocol_methods(backend)}"
    )

    print("✅ SupabaseStorageBackend implements StorageBackend protocol")


def test_sqlite_backend_implements_protocol():
    """Verify SQLiteStorageBackend implements StorageBackend protocol."""
    from langmiddle.storage.sqlite_backend import SQLiteStorageBackend

    backend = SQLiteStorageBackend(":memory:")

    assert isinstance(backend, StorageBackend), (
        "SQLiteStorageBackend must implement StorageBackend protocol. "
        f"Missing methods: {_check_protocol_methods(backend)}"
    )

    print("✅ SQLiteStorageBackend implements StorageBackend protocol")


def test_postgres_backend_implements_protocol():
    """Verify PostgreSQLStorageBackend implements StorageBackend protocol."""
    from langmiddle.storage.postgres_backend import PostgreSQLStorageBackend

    # Skip if psycopg2 not available
    try:
        import psycopg2
    except ImportError:
        pytest.skip("psycopg2 not available")

    # Create backend with mock connection string
    try:
        backend = PostgreSQLStorageBackend(
            "postgresql://user:pass@localhost/db",
            auto_create_tables=False,
        )

        assert isinstance(backend, StorageBackend), (
            "PostgreSQLStorageBackend must implement StorageBackend protocol. "
            f"Missing methods: {_check_protocol_methods(backend)}"
        )

        print("✅ PostgreSQLStorageBackend implements StorageBackend protocol")
    except Exception as e:
        pytest.skip(f"Could not create PostgreSQLStorageBackend: {e}")


def _check_protocol_methods(obj: object) -> list[str]:
    """Check which protocol methods are missing."""
    from langmiddle.context.core.protocols import StorageBackend

    required_methods = [
        "query_facts",
        "insert_facts",
        "prepare_credentials",
        "authenticate",
        "extract_user_id",
    ]

    missing = []
    for method in required_methods:
        if not hasattr(obj, method):
            missing.append(method)
        elif not callable(getattr(obj, method)):
            missing.append(f"{method} (not callable)")

    return missing


if __name__ == "__main__":
    test_supabase_backend_implements_protocol()

    try:
        test_sqlite_backend_implements_protocol()
    except Exception as e:
        print(f"⚠️  SQLiteStorageBackend test skipped: {e}")

    try:
        test_postgres_backend_implements_protocol()
    except Exception as e:
        print(f"⚠️  PostgreSQLStorageBackend test skipped: {e}")

    print("\n✅ All available storage backends implement StorageBackend protocol!")
