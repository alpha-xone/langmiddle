"""
SQLite inspection tools for LangMiddle.

These helper functions allow developers to inspect the contents of the local SQLite database,
including virtual embedding tables which are otherwise hard to view.
"""

import sqlite3
from typing import Any, Dict, List, Optional


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Get a connection to the SQLite database with sqlite-vec loaded if available."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (ImportError, AttributeError):
        # sqlite-vec might not be installed or loadable
        pass

    return conn


def inspect_database(db_path: str) -> Dict[str, int]:
    """
    Get a summary of all tables and their row counts.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Dictionary mapping table names to row counts.
    """
    summary = {}
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            try:
                # For virtual tables, COUNT(*) might be slow or behave differently, but usually works
                count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                summary[table] = count
            except sqlite3.Error:
                summary[table] = -1     # Error reading table
    finally:
        conn.close()
    return summary


def peek_table(
    db_path: str,
    table_name: str,
    limit: int = 5,
    columns: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Peek into any table.

    Args:
        db_path: Path to the SQLite database file.
        table_name: Name of the table to inspect.
        limit: Number of rows to return.
        columns: Specific columns to select (defaults to *).
    """
    conn = _get_connection(db_path)
    try:
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table_name} LIMIT ?"
        cursor = conn.execute(query, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"Error reading table {table_name}: {e}")
        return []
    finally:
        conn.close()


def peek_embeddings(
    db_path: str,
    dimension: int = 1024,
    limit: int = 5,
    include_vector: bool = False
) -> List[Dict[str, Any]]:
    """
    Peek into the virtual embedding table.

    Args:
        db_path: Path to the SQLite database file.
        dimension: Dimension of the embeddings (suffix of table name).
        limit: Number of rows to return.
        include_vector: Whether to retrieve the actual vector data (can be large).
    """
    table_name = f"fact_embeddings_{dimension}"
    conn = _get_connection(db_path)
    try:
        # Check if table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        if not cursor.fetchone():
            print(f"Table {table_name} not found.")
            return []

        # Construct query
        cols = "rowid, fact_id, user_id"
        if include_vector:
            # Use vec_to_json if available to make it readable, otherwise just select the blob
            try:
                conn.execute("SELECT vec_to_json(?)", (b'\x00' * 4,))  # Test if function exists
                cols += ", vec_to_json(embedding) as embedding"
            except sqlite3.Error:
                cols += ", embedding"

        query = f"SELECT {cols} FROM {table_name} LIMIT ?"
        cursor = conn.execute(query, (limit,))

        results = []
        for row in cursor.fetchall():
            item = dict(row)
            results.append(item)

        return results
    except sqlite3.Error as e:
        print(f"Error reading embeddings: {e}")
        return []
    finally:
        conn.close()
