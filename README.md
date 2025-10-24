# langmiddle

Middlewares for LangChain / LangGraph

[![PyPI version](https://badge.fury.io/py/langmiddle.svg)](https://badge.fury.io/py/langmiddle)
[![Python versions](https://img.shields.io/pypi/pyversions/langmiddle.svg)](https://pypi.org/project/langmiddle/)
[![License](https://img.shields.io/github/license/alpha-xone/langmiddle.svg)](https://github.com/alpha-xone/langmiddle/blob/main/LICENSE)

## Overview

Production-ready middleware for **LangChain** and **LangGraph v1** with multi-backend chat history persistence. Store conversations in SQLite, Supabase, or Firebase with zero configuration required.

**Key Features:**
- ✅ **LangChain/LangGraph v1 Compatible**: Native middleware pattern support
- � **Zero Config Start**: Defaults to in-memory SQLite—no setup needed
- 🔄 **Multi-Backend Storage**: Switch between SQLite, Supabase, Firebase with one parameter
- 🔒 **Production Ready**: JWT authentication, RLS support, type-safe

## Installation

**Core Package** (SQLite only):
```bash
pip install langmiddle
```

**With Optional Backends:**
```bash
# For Supabase support
pip install langmiddle[supabase]

# For Firebase support
pip install langmiddle[firebase]

# All backends
pip install langmiddle[all]
```

## Quick Start - LangChain Middleware

```python
from langmiddle import ChatSaver, ToolFilter

# Initialize middleware with desired backend
agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ToolFilter(),
        ChatSaver(backend="sqlite", db_path="./chat_history.db")
    ],
)

agent.invoke(
    input={"messages": [{"role": "user", "content": "Hello!"}]},
    context={
        "user_id": "user-123",
        "session_id": "session-123",
    }
)
```

## Storage Backends

| Backend  | Use Case | Pros | Cons | Setup |
|----------|----------|------|------|-------|
| **SQLite** | Development, Single-user | Simple, Local, Fast, No setup | Not distributed | None |
| **Supabase** | Production Web Apps | Scalable, Real-time, RLS, Multi-user | Requires configuration | Environment vars |
| **Firebase** | Mobile, Google ecosystem | Real-time, Managed, Global | Google-specific | Service account |

### SQLite Configuration

```python
# Local file
backend_type="sqlite", db_path="./chat.db"

# In-memory (testing)
backend_type="sqlite", db_path=":memory:"
```

### Supabase Configuration

```bash
# .env file or environment variables
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
```

### Firebase Configuration

```python
# Service account credentials file
backend_type="firebase", credentials_path="./firebase-creds.json"

# Or use GOOGLE_APPLICATION_CREDENTIALS environment variable
```
