# PostgreSQL Backend Setup

This guide explains how to set up and use the PostgreSQL storage backend with langmiddle.

## Overview

The PostgreSQL backend provides direct PostgreSQL database integration for storing chat history. It's ideal for:

- Applications with existing PostgreSQL infrastructure
- Self-hosted deployments
- Scenarios requiring full database control
- Multi-tenant applications with custom user management

## Comparison: PostgreSQL vs Supabase

Both backends share the same underlying PostgreSQL database schema, but differ in how they connect:

| Feature | PostgreSQL Backend | Supabase Backend |
|---------|-------------------|------------------|
| Connection | Direct psycopg2 connection | Supabase REST API |
| Authentication | Connection string | JWT tokens + Row Level Security |
| User Management | Custom (string user_id) | Supabase Auth (UUID) |
| Dependencies | psycopg2-binary | supabase, python-jose |
| Best For | Self-hosted, custom auth | Managed hosting, Supabase ecosystem |

## Installation

Install langmiddle with PostgreSQL support:

```bash
pip install langmiddle[postgres]
```

Or install the dependencies manually:

```bash
pip install psycopg2-binary python-dotenv
```

## Database Setup

### 1. Create PostgreSQL Database

```bash
# Using psql
createdb chatdb

# Or in PostgreSQL
CREATE DATABASE chatdb;
```

### 2. Automatic Table Creation

The backend can automatically create the required tables:

```python
from langmiddle import ChatStorage

storage = ChatStorage.create(
    "postgres",
    connection_string="postgresql://user:password@localhost:5432/chatdb",
    auto_create_tables=True  # Creates tables if they don't exist
)
```

### 3. Manual Table Creation (Optional)

If you prefer to create tables manually, use the SQL files in `langmiddle/storage/postgres/`:

```sql
-- Create trigger function first
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Then run chat_threads.sql and chat_messages.sql
```

## Configuration

### Connection String Format

```
postgresql://[user]:[password]@[host]:[port]/[database]
```

Example:
```
postgresql://myuser:mypassword@localhost:5432/chatdb
```

### Using Environment Variables

Create a `.env` file in your project root:

```env
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/chatdb
```

Then use the backend without explicitly passing the connection string:

```python
from langmiddle import ChatStorage

# Automatically loads from .env
storage = ChatStorage.create("postgres")
```

## Usage Examples

### Basic Usage

```python
from langmiddle import ChatStorage
from langchain_core.messages import HumanMessage, AIMessage

# Create storage
storage = ChatStorage.create(
    "postgres",
    connection_string="postgresql://user:password@localhost:5432/chatdb"
)

# Create messages
messages = [
    HumanMessage(content="Hello!", id="msg-1"),
    AIMessage(content="Hi there!", id="msg-2"),
]

# Save messages
result = storage.save_chat_history(
    thread_id="thread-123",
    credentials={"user_id": "user-456"},
    messages=messages
)

print(f"Saved {result['saved_count']} messages")
```

### Incremental Saves (Avoiding Duplicates)

```python
# First save
result1 = storage.save_chat_history(
    thread_id="thread-123",
    credentials={"user_id": "user-456"},
    messages=first_batch
)

saved_msg_ids = result1['saved_msg_ids']

# Second save - pass saved_msg_ids to skip duplicates
result2 = storage.save_chat_history(
    thread_id="thread-123",
    credentials={"user_id": "user-456"},
    messages=second_batch,
    saved_msg_ids=saved_msg_ids  # Reuse from previous save
)

print(f"Skipped {result2['skipped_count']} duplicate messages")
```

### LangGraph Integration

```python
from langgraph.checkpoint.base import BaseCheckpointSaver
from langmiddle import ChatStorage

# Initialize storage
storage = ChatStorage.create("postgres")

# Use in your LangGraph workflow
def save_to_postgres(state):
    messages = state.get("messages", [])
    storage.save_chat_history(
        thread_id=state["thread_id"],
        credentials={"user_id": state["user_id"]},
        messages=messages
    )
    return state
```

## Schema Details

### chat_threads Table

```sql
- id (text, primary key)
- user_id (text)
- title (text, default '')
- metadata (jsonb)
- created_at (timestamp with time zone)
- updated_at (timestamp with time zone)
```

### chat_messages Table

```sql
- id (text, primary key)
- user_id (text)
- thread_id (text, foreign key to chat_threads)
- content (text)
- role (text: 'user', 'human', 'assistant', 'ai', 'tool', 'system')
- metadata (jsonb)
- usage_metadata (jsonb, nullable)
- created_at (timestamp with time zone)
```

## Key Differences from Supabase Schema

1. **User ID Type**: Uses `text` instead of `uuid` for flexibility
2. **No Foreign Key to auth.users**: Manages users independently
3. **Simpler Authentication**: No Row Level Security (RLS) policies

## Migration from Supabase

If you're migrating from Supabase to direct PostgreSQL:

1. The data schema is compatible (just different user_id types)
2. You can use the same PostgreSQL database
3. Update connection method from Supabase client to direct connection
4. Remove JWT authentication logic
5. Handle user_id as string instead of UUID

## Performance Considerations

- **Connection Pooling**: Built-in connection pool (1-10 connections)
- **Batch Operations**: Use incremental saves with `saved_msg_ids`
- **Indexes**: Automatically created on user_id, thread_id, and created_at
- **JSONB**: Metadata stored as JSONB for efficient querying

## Troubleshooting

### Connection Issues

```python
# Test connection
import psycopg2
conn = psycopg2.connect("postgresql://user:password@localhost:5432/chatdb")
print("Connection successful!")
conn.close()
```

### Table Creation Fails

Ensure you have CREATE TABLE privileges:

```sql
GRANT CREATE ON DATABASE chatdb TO myuser;
```

### Import Error

```bash
# If you get "ImportError: No module named 'psycopg2'"
pip install psycopg2-binary
```

## Advanced Configuration

### Custom Connection Pool

The backend uses a connection pool (1-10 connections by default). To modify, edit `postgres_backend.py`:

```python
self._connection_pool = pool.SimpleConnectionPool(
    5, 20,  # min=5, max=20 connections
    connection_string
)
```

### SSL Connections

For secure connections, add SSL parameters to your connection string:

```
postgresql://user:password@host:5432/db?sslmode=require
```

## See Also

- [Supabase Setup Guide](./SUPABASE_SETUP.md) - For Supabase backend
- [PostgreSQL Usage Example](../examples/postgres_usage.py)
- [Multi-Backend Usage](../examples/multi_backend_usage.py)
