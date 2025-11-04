# 🧩 LangMiddle — Composable Middlewares for LangGraph

> Lightweight, plug‑and‑play middlewares for LangGraph developers.

[![CI](https://github.com/alpha-xone/langmiddle/actions/workflows/ci.yml/badge.svg)](https://github.com/alpha-xone/langmiddle/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/langmiddle.svg)](https://pypi.org/project/langmiddle/)
[![Python versions](https://img.shields.io/pypi/pyversions/langmiddle.svg)](https://pypi.org/project/langmiddle/)
[![License](https://img.shields.io/github/license/alpha-xone/langmiddle.svg)](https://github.com/alpha-xone/langmiddle/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/alpha-xone/langmiddle?style=social)](https://github.com/alpha-xone/langmiddle)

## Overview

Production-ready middleware for **LangChain** and **LangGraph v1** with multi-backend chat history persistence. Store conversations in SQLite, Supabase, or Firebase with zero configuration required.

**Key Features:**
- ✅ **LangChain / LangGraph v1 Compatible**: Native middleware pattern support
- ⚡ **Zero Config Start**: Defaults to in-memory SQLite—no setup needed
- 🔄 **Multi-Backend Storage**: Switch between SQLite, PostgreSQL, Supabase, Firebase with one parameter
- 🔒 **Production Ready**: JWT authentication, RLS support, type-safe
 - 🧠 **Context Engineering**: Built-in support for long-term facts, semantic memories, and context management

## Available middleware

| Middleware | Description | Supported Backends |
|---|---|---|
| ToolRemover | Removes tool-related messages from the conversation state (pre/post agent). | N/A (no backend needed) |
| ChatSaver | Persists chat histories | SQLite ✅, PostgreSQL ✅, Supabase ✅, Firebase ✅ |
| ContextEngineer | Auto context management with semantic memories | Supabase 🚧 (in progress) |

## Installation

**Core Package** (SQLite only):
```bash
pip install langmiddle
```

**With Optional Backends:**
```bash
# For PostgreSQL support
pip install langmiddle[postgres]

# For Supabase support
pip install langmiddle[supabase]

# For Firebase support
pip install langmiddle[firebase]

# All backends
pip install langmiddle[all]
```

## Quick Start - LangChain Middleware

```python
from langmiddle.history import ChatSaver, ToolRemover, StorageContext

# Initialize middleware with desired backend
agent = create_agent(
    model="gpt-4o",
    tools=[],
    context_schema=StorageContext,
    middleware=[
        ToolRemover(),
        ChatSaver(backend="sqlite", db_path="./chat_history.db")
    ],
)

agent.invoke(
    input={"messages": [{"role": "user", "content": "Hello!"}]},
    context={
        "user_id": "***",           # User ID in UUID format
        "session_id": "***",        # Thread ID in UUID format
    },
)
```

## ContextEngineer

ContextEngineer provides automatic context and memory management for conversations. It can store, retrieve, and query long-term facts (semantic memories) using the project's supported storage backends.

```python
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langmiddle.context import ContextEngineer

# Initialize model and embedder
llm = init_chat_model("gpt-4o")
embedder = init_embeddings("text-embedding-3-small")

# When running for **the first time**, create databases in the backend
# Requires connection string for table creations
store = ChatStorage.create(
    "supabase",
    auto_create_tables=True,
    enable_facts=True,
)

# Use in your agent
agent = create_agent(
    model="gpt-4o",
    tools=[],
    context_schema=StorageContext,
    middleware=[
        ContextEngineer(
            model=llm,
            embedder=embedder,
            backend="supabase",
            backend_kwargs={'enable_facts': True},
        ),
    ],
)

agent.invoke(
    input={"messages": [{"role": "user", "content": "Tell me about my preferences."}]},
    context={
        "user_id": "***",           # User ID in UUID format
        "session_id": "***",        # Thread ID in UUID format
        "auth_token": "***",        # JWT token for supabase backend
    },
)
```

## Storage Backends

| Backend  | Use Case | Pros | Cons | Setup |
|----------|----------|------|------|-------|
| **SQLite** | Development, Single-user | Simple, Local, Fast, No setup | Not distributed | None |
| **PostgreSQL** | Self-hosted, Custom auth | Full control, Standard SQL, Flexible | Manual setup | Connection string |
| **Supabase** | Production Web Apps | Managed PostgreSQL, RLS, Real-time | Supabase-specific | Environment vars |
| **Firebase** | Mobile, Google ecosystem | Real-time, Managed, Global | Google-specific | Service account |

### SQLite Configuration

```python
# Local file
backend_type="sqlite", db_path="./chat.db"

# In-memory (testing)
backend_type="sqlite", db_path=":memory:"
```

### PostgreSQL Configuration

```bash
# .env file or environment variables
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/dbname
```

```python
# Or pass directly
backend_type="postgres",
connection_string="postgresql://user:password@localhost:5432/dbname",
auto_create_tables=True
```

See [PostgreSQL Setup Guide](docs/POSTGRES_SETUP.md) for details.

### Supabase Configuration

```bash
# .env file or environment variables
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key

# To create tables
SUPABASE_CONNECTION_STRING=your_connection_string
```

### Firebase Configuration

```python
# Service account credentials file
backend_type="firebase", credentials_path="./firebase-creds.json"

# Or use GOOGLE_APPLICATION_CREDENTIALS environment variable
```

---

## Contributing

We welcome contributions! If LangMiddle helped you, please ⭐️ the repo to help others discover it.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
