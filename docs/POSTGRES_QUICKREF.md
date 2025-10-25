# PostgreSQL Backend - Quick Reference

## What Was Added

✅ Direct PostgreSQL backend support
✅ Shared code architecture (PostgreSQLBaseBackend)
✅ Refactored Supabase to reuse PostgreSQL logic
✅ Standalone PostgreSQL SQL schemas (no Supabase dependencies)
✅ Connection pooling
✅ Complete documentation and examples

## Quick Start

### Installation
```bash
pip install langmiddle[postgres]
```

### Basic Usage
```python
from langmiddle.storage import ChatStorage

# Option 1: Direct connection string
storage = ChatStorage.create(
    "postgres",
    connection_string="postgresql://user:password@localhost:5432/chatdb",
    auto_create_tables=True
)

# Option 2: From environment variable
# POSTGRES_CONNECTION_STRING=postgresql://...
storage = ChatStorage.create("postgres")

# Save messages
result = storage.save_chat_history(
    thread_id="thread-123",
    credentials={"user_id": "user-456"},
    messages=messages
)
```

## Architecture Diagram

```
Before:
├── base.py (Abstract)
├── supabase_backend.py (Full implementation with SQL)
├── sqlite_backend.py
└── firebase_backend.py

After:
├── base.py (Abstract)
├── postgres_base.py (NEW - Shared PostgreSQL logic)
│   ├── Table creation
│   ├── SQL queries
│   └── Common operations
├── postgres_backend.py (NEW - Direct PostgreSQL)
│   ├── psycopg2 connection pool
│   └── Implements _execute_query()
├── supabase_backend.py (REFACTORED - Inherits from postgres_base)
│   ├── Supabase client
│   ├── JWT authentication
│   └── REST API wrapper
├── sqlite_backend.py
└── firebase_backend.py
```

## Key Benefits

| Aspect | Benefit |
|--------|---------|
| **Code Reuse** | ~150 lines of shared logic |
| **Maintainability** | Single source of truth for PostgreSQL operations |
| **Flexibility** | Choose between managed (Supabase) or self-hosted (PostgreSQL) |
| **Compatibility** | Same schema, easy migration |
| **No Breaking Changes** | Existing Supabase code works unchanged |

## When to Use Each Backend

```
SQLite          → Development, testing, single-user
PostgreSQL      → Self-hosted, custom auth, full control
Supabase        → Managed hosting, built-in auth, RLS
Firebase        → Mobile apps, Google ecosystem
```

## Files Changed/Added

### New Files
- `langmiddle/storage/postgres_base.py` - Shared PostgreSQL logic
- `langmiddle/storage/postgres_backend.py` - Direct PostgreSQL backend
- `langmiddle/storage/postgres/chat_threads.sql` - PostgreSQL schema
- `langmiddle/storage/postgres/chat_messages.sql` - PostgreSQL schema
- `examples/postgres_usage.py` - Usage examples
- `docs/POSTGRES_SETUP.md` - Setup guide
- `docs/POSTGRES_IMPLEMENTATION.md` - Implementation details

### Modified Files
- `langmiddle/storage/supabase_backend.py` - Now inherits from postgres_base
- `langmiddle/storage/__init__.py` - Added postgres backend
- `pyproject.toml` - Added postgres optional dependencies
- `MANIFEST.in` - Include SQL files in package
- `README.md` - Added PostgreSQL documentation

## Schema Comparison

### PostgreSQL Schema
```sql
-- chat_threads
id          text        -- Flexible string IDs
user_id     text        -- No foreign key constraint
thread_id   text
```

### Supabase Schema
```sql
-- chat_threads
id          uuid        -- UUID with auth.users FK
user_id     uuid        -- References auth.users(id)
thread_id   uuid
```

## Testing

```bash
# Test PostgreSQL backend
python examples/postgres_usage.py

# Test that Supabase still works
python examples/multi_backend_usage.py
```

## Next Steps

1. **Deploy**: Use in your application
2. **Migrate**: Move from Supabase to PostgreSQL or vice versa
3. **Extend**: Add custom authentication logic
4. **Scale**: Tune connection pool settings

## Documentation

- [Setup Guide](POSTGRES_SETUP.md) - Complete setup instructions
- [Implementation Details](POSTGRES_IMPLEMENTATION.md) - Architecture explanation
- [Example Code](../examples/postgres_usage.py) - Working examples
