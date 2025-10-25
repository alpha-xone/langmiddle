# PostgreSQL Backend Implementation Summary

## Overview

Added direct PostgreSQL backend support to langmiddle, with code reuse between PostgreSQL and Supabase backends since Supabase is built on PostgreSQL.

## Architecture

```
ChatStorageBackend (Abstract Base)
         ↑
         |
PostgreSQLBaseBackend (Shared PostgreSQL Logic)
         ↑
         |
    ┌────┴────┐
    |         |
PostgreSQL  Supabase
Backend     Backend
```

## Code Reuse Strategy

### PostgreSQLBaseBackend (`postgres_base.py`)
**Shared functionality:**
- Table creation logic using psycopg2
- SQL-based message saving
- SQL-based message ID retrieval
- Thread existence checking
- Common error handling

### SupabaseStorageBackend (`supabase_backend.py`)
**Supabase-specific:**
- Supabase client initialization
- JWT authentication
- REST API query builder
- Row Level Security (RLS) support
- UUID-based user IDs

### PostgreSQLStorageBackend (`postgres_backend.py`)
**PostgreSQL-specific:**
- Direct psycopg2 connection pool
- Raw SQL queries via `_execute_query()`
- String-based user IDs
- No authentication layer (handled by connection string)

## File Structure

```
langmiddle/storage/
├── base.py                    # Abstract interface
├── postgres_base.py           # NEW: Shared PostgreSQL logic
├── postgres_backend.py        # NEW: Direct PostgreSQL implementation
├── supabase_backend.py        # MODIFIED: Now inherits from postgres_base
├── sqlite_backend.py          # Unchanged
├── firebase_backend.py        # Unchanged
├── __init__.py               # MODIFIED: Added postgres backend
├── postgres/                  # NEW: PostgreSQL SQL schemas
│   ├── chat_threads.sql
│   └── chat_messages.sql
└── supabase/                  # Existing Supabase SQL schemas
    ├── chat_threads.sql
    └── chat_messages.sql
```

## Key Differences: PostgreSQL vs Supabase Schemas

| Aspect | PostgreSQL | Supabase |
|--------|-----------|----------|
| **user_id type** | `text` | `uuid` |
| **auth.users FK** | None | `REFERENCES auth.users` |
| **thread_id type** | `text` | `uuid` |
| **Authentication** | Connection string | JWT tokens |
| **Row Level Security** | Not used | RLS policies |
| **Use Case** | Self-hosted, custom auth | Managed, Supabase ecosystem |

## Installation

```bash
# PostgreSQL only
pip install langmiddle[postgres]

# Supabase (includes PostgreSQL dependencies)
pip install langmiddle[supabase]
```

## Usage Comparison

### PostgreSQL
```python
from langmiddle import ChatStorage

storage = ChatStorage.create(
    "postgres",
    connection_string="postgresql://user:pass@host:5432/db",
    auto_create_tables=True
)

result = storage.save_chat_history(
    thread_id="thread-123",
    credentials={"user_id": "user-456"},  # Simple string user_id
    messages=messages
)
```

### Supabase
```python
storage = ChatStorage.create(
    "supabase",
    supabase_url="https://xxx.supabase.co",
    supabase_key="anon-key",
    auto_create_tables=True
)

result = storage.save_chat_history(
    thread_id="thread-123",
    credentials={"jwt_token": "..."},  # JWT for RLS
    messages=messages
)
```

## Benefits of Code Reuse

1. **Reduced Duplication**: ~150 lines of shared code moved to base class
2. **Consistent Behavior**: Same SQL logic for both backends
3. **Easier Maintenance**: Bug fixes apply to both backends
4. **Clear Separation**: Backend-specific code is isolated
5. **Future Extensions**: Easy to add more PostgreSQL-based backends

## Testing Checklist

- [x] PostgreSQL backend creates tables automatically
- [x] PostgreSQL backend saves messages correctly
- [x] PostgreSQL backend handles duplicates
- [x] Supabase backend still works after refactoring
- [x] Connection pooling works
- [x] Error handling works
- [x] SQL injection protection (parameterized queries)
- [x] Package includes SQL files in distribution

## Documentation

- [PostgreSQL Setup Guide](POSTGRES_SETUP.md) - Complete setup instructions
- [Example Usage](../examples/postgres_usage.py) - Code examples
- [README.md](../README.md) - Updated with PostgreSQL backend

## Migration Path

**From Supabase to PostgreSQL:**
1. Same database, different connection method
2. Update user_id handling (UUID → string)
3. Remove JWT authentication
4. Update foreign key constraints

**From PostgreSQL to Supabase:**
1. Add Supabase Auth
2. Convert user_id to UUID
3. Enable Row Level Security
4. Use Supabase client

## Performance Considerations

- **Connection Pooling**: PostgreSQL backend uses 1-10 connections by default
- **Batch Operations**: Both backends support incremental saves
- **Indexes**: Same index strategy for both backends
- **JSONB**: Efficient metadata storage in both

## Future Enhancements

Possible improvements:
1. **AsyncIO support**: Async versions of both backends
2. **SQLAlchemy integration**: ORM support
3. **Migration tools**: Automated migration between backends
4. **Connection pool configuration**: Expose pool settings
5. **Read operations**: Implement message retrieval methods
