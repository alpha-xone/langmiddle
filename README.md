# 🧩 LangMiddle — Production Middleware for LangGraph

> **Supercharge your LangGraph agents with plug‑and‑play memory, context management, and chat persistence.**

[![CI](https://github.com/alpha-xone/langmiddle/actions/workflows/ci.yml/badge.svg)](https://github.com/alpha-xone/langmiddle/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/langmiddle.svg)](https://pypi.org/project/langmiddle/)
[![Python versions](https://img.shields.io/pypi/pyversions/langmiddle.svg)](https://pypi.org/project/langmiddle/)
[![License](https://img.shields.io/github/license/alpha-xone/langmiddle.svg)](https://github.com/alpha-xone/langmiddle/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/alpha-xone/langmiddle?style=social)](https://github.com/alpha-xone/langmiddle)

---

## 🎯 Why LangMiddle?

Building production LangGraph agents? You need:
- 💾 **Persistent chat history** across sessions
- 🧠 **Long-term memory** that remembers user preferences and context
- 🔍 **Semantic fact retrieval** to inject relevant knowledge
- 🗑️ **Clean message handling** without tool noise

LangMiddle delivers all of this with **zero boilerplate**—just add middleware to your agent.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🚀 Zero Config Start** | Works out-of-the-box with in-memory SQLite—no database setup |
| **🔄 Multi-Backend Storage** | Switch between SQLite, PostgreSQL, Supabase, Firebase with one parameter |
| **🧠 Semantic Memory** | Automatic fact extraction, deduplication, and context injection |
| **📝 Smart Summarization** | Auto-compress long conversations while preserving context |
| **🔐 Production Ready** | JWT auth, RLS support, type-safe APIs, comprehensive logging |
| **⚡ LangGraph Native** | Built for LangChain/LangGraph v1 middleware pattern |

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Available Middleware](#-available-middleware)
  - [ChatSaver](#chatsaver---persist-conversations)
  - [ToolRemover](#toolremover---clean-tool-messages)
  - [ContextEngineer](#contextengineer---intelligent-memory--context)
- [Storage Backends](#-storage-backends)
- [Examples](#-examples)
- [Contributing](#-contributing)

---

## 📦 Installation

**Core Package** (includes SQLite support):
```bash
pip install langmiddle
```

**With Optional Backends:**
```bash
# For PostgreSQL
pip install langmiddle[postgres]

# For Supabase (includes PostgreSQL)
pip install langmiddle[supabase]

# For Firebase
pip install langmiddle[firebase]

# All backends + extras
pip install langmiddle[all]
```

---

## 🚀 Quick Start

### Basic Chat Persistence (SQLite)

Get started in 30 seconds:

```python
from langchain.agents import create_agent
from langmiddle.history import ChatSaver, StorageContext

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    context_schema=StorageContext,
    middleware=[
        ChatSaver()  # Uses in-memory SQLite by default
    ],
)

# Chat history automatically saved!
agent.invoke(
    input={"messages": [{"role": "user", "content": "Hello!"}]},
    context=StorageContext(
        thread_id="conversation-123",
        user_id="user-456"
    )
)
```

### With Long-Term Memory (ContextEngineer)

Add semantic memory and auto-summarization:

```python
from langmiddle.context import ContextEngineer

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    context_schema=StorageContext,
    middleware=[
        ContextEngineer(
            model="openai:gpt-4o",
            embedder="openai:text-embedding-3-small",
            backend="supabase",
            backend_kwargs={'enable_facts': True}
        )
    ],
)

# Agent now remembers user preferences, past decisions, and context!
agent.invoke(
    input={"messages": [{"role": "user", "content": "What are my food preferences?"}]},
    context=StorageContext(
        thread_id="conversation-123",
        user_id="user-456",
        auth_token="your-jwt-token"
    )
)
```

---

## 🛠️ Available Middleware

### ChatSaver - Persist Conversations

**Automatically save** chat histories to your database of choice.

**Features:**
- ✅ Multi-backend: SQLite, PostgreSQL, Supabase, Firebase
- ✅ Automatic deduplication (skips already-saved messages)
- ✅ Save interval control (every N turns)
- ✅ Custom state persistence

**Example:**
```python
from langmiddle.history import ChatSaver

ChatSaver(
    backend="sqlite",
    db_path="./chat.db",
    save_interval=1  # Save after every agent response
)
```

**Supported Backends:**
| Backend | Use Case | Auth Required |
|---------|----------|---------------|
| SQLite | Development, local apps | ❌ No |
| PostgreSQL | Self-hosted production | ❌ No |
| Supabase | Managed PostgreSQL + RLS | ✅ JWT |
| Firebase | Mobile, real-time apps | ✅ ID token |

---

### ToolRemover - Clean Tool Messages

**Remove tool-related clutter** from conversation history.

**Why?** Tool call messages and responses bloat chat history and aren't relevant for long-term storage.

**Example:**
```python
from langmiddle.history import ToolRemover

middleware=[
    ToolRemover(when="both"),  # Filter before AND after agent
    ChatSaver()  # Clean messages are saved
]
```

**Options:**
- `when="before"` - Filter before agent sees messages
- `when="after"` - Filter before saving to storage
- `when="both"` - Filter in both directions (recommended)

---

### ContextEngineer - Intelligent Memory & Context

**The brain of your agent** — automatic fact extraction, semantic search, and context injection.

#### 🧠 What It Does

1. **Extracts Facts**: Identifies user preferences, goals, and key information
2. **Stores Semantically**: Embeds facts for similarity search
3. **Retrieves Contextually**: Injects relevant memories based on user queries
4. **Auto-Summarizes**: Compresses old conversations to save tokens

#### 🔥 Key Features

| Feature | Description |
|---------|-------------|
| **Semantic Fact Storage** | Vector-based storage with deduplication |
| **Smart Extraction** | Filters out ephemeral states (e.g., "user understood") |
| **Namespace Organization** | Hierarchical fact categories (`["user", "preferences", "food"]`) |
| **Automatic Summarization** | Configurable token thresholds |
| **Atomic Query Breaking** | Splits complex queries for better retrieval |
| **Relevance Scoring** | Dynamic scoring based on recency, access patterns, and usage feedback |
| **Adaptive Formatting** | Context detail adjusts based on fact relevance |
| **Caching** | Embeddings and core facts cached for performance |

#### 📝 Example Usage

```python
from langmiddle.context import ContextEngineer
from langmiddle.storage import ChatStorage

# 1. First-time setup: Create tables
store = ChatStorage.create(
    "supabase",
    auto_create_tables=True,
    enable_facts=True
)

# 2. Initialize middleware
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    context_schema=StorageContext,
    middleware=[
        ContextEngineer(
            model="openai:gpt-4o",
            embedder="openai:text-embedding-3-small",
            backend="supabase",
            max_tokens_before_summarization=5000,  # Trigger summarization at 5k tokens
            extraction_interval=3,  # Extract facts every 3 turns
            backend_kwargs={'enable_facts': True}
        )
    ],
)

# 3. Use it!
response = agent.invoke(
    input={"messages": [{"role": "user", "content": "I love spicy Thai food"}]},
    context=StorageContext(
        thread_id="conversation-123",
        user_id="user-456",
        auth_token="your-jwt-token"
    )
)

# Later in the conversation...
response = agent.invoke(
    input={"messages": [{"role": "user", "content": "Recommend a restaurant"}]},
    context=StorageContext(
        thread_id="conversation-123",
        user_id="user-456",
        auth_token="your-jwt-token"
    )
)
# Agent remembers: "user loves spicy Thai food" and uses it for recommendations!
```

#### ⚙️ Configuration Options

```python
ContextEngineer(
    model="openai:gpt-4o",                    # LLM for extraction & summarization
    embedder="openai:text-embedding-3-small", # Embedding model for semantic search
    backend="supabase",                       # Storage backend

    # Extraction settings
    extraction_interval=3,                    # Extract facts every N turns
    max_tokens_before_extraction=None,        # Or trigger by token count

    # Summarization settings
    max_tokens_before_summarization=5000,     # Auto-summarize at 5k tokens

    # Context injection
    core_namespaces=[                         # Always-loaded fact categories
        ["user", "personal_info"],
        ["user", "preferences"]
    ],

    # Relevance scoring (Phase 3)
    relevance_threshold=0.3,                  # Minimum relevance score to inject
    similarity_weight=0.7,                    # Weight for vector similarity
    relevance_weight=0.3,                     # Weight for relevance score
    enable_adaptive_formatting=True,          # Adjust detail based on relevance

    # Backend configuration
    backend_kwargs={'enable_facts': True}
)
```

#### 📊 What Gets Stored

**Facts Examples:**
```json
[
  {
    "content": "User prefers concise and formal answers",
    "namespace": ["user", "preferences", "communication"],
    "intensity": 1.0,
    "confidence": 0.97,
    "language": "en"
  },
  {
    "content": "User's name is Alex",
    "namespace": ["user", "personal_info"],
    "intensity": 0.9,
    "confidence": 0.98,
    "language": "en"
  }
]
```

**What's NOT stored** (filtered out by design):
- ❌ Transient states: "User understands X", "User is satisfied"
- ❌ Single-use requests: "User wants a code example"
- ❌ Politeness markers: "User says thank you"
- ❌ Momentary emotions: "User feels frustrated right now"

---

## 💾 Storage Backends

### Comparison Guide

| Backend | Best For | Setup Complexity | Scalability | Auth | Cost |
|---------|----------|------------------|-------------|------|------|
| **SQLite** | • Local development<br>• Demos<br>• Single-user apps | ⭐ Trivial | 🔵 Single machine | None | Free |
| **PostgreSQL** | • Self-hosted production<br>• Custom infrastructure<br>• Full control | ⭐⭐ Medium | 🔵🔵🔵 High (with replication) | Custom | Infrastructure cost |
| **Supabase** | • Web apps<br>• Multi-tenant SaaS<br>• Real-time features | ⭐⭐ Easy | 🔵🔵🔵 High (managed) | JWT + RLS | Free tier + usage |
| **Firebase** | • Mobile apps<br>• Google Cloud ecosystem<br>• Real-time sync | ⭐⭐ Easy | 🔵🔵🔵 Global (managed) | Firebase Auth | Free tier + usage |

---

### 🗂️ Backend Configuration

<details>
<summary><b>SQLite</b> — Zero-config local storage</summary>

```python
from langmiddle.history import ChatSaver

# File-based (persistent)
ChatSaver(backend="sqlite", db_path="./chat.db")

# In-memory (testing/dev)
ChatSaver(backend="sqlite", db_path=":memory:")
```

**No environment variables needed!**

</details>

<details>
<summary><b>PostgreSQL</b> — Self-hosted database</summary>

**Environment variables** (`.env`):
```bash
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/dbname
```

**Python code:**
```python
from langmiddle.storage import ChatStorage

# First-time setup: create tables
store = ChatStorage.create(
    "postgres",
    connection_string="postgresql://user:pass@localhost:5432/db",
    auto_create_tables=True
)

# In middleware
ChatSaver(backend="postgres")
```

📖 [PostgreSQL Setup Guide](docs/POSTGRES_SETUP.md)

</details>

<details>
<summary><b>Supabase</b> — Managed PostgreSQL with RLS</summary>

**Environment variables** (`.env`):
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key

# For table creation (one-time):
SUPABASE_CONNECTION_STRING=postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres
```

**Python code:**
```python
from langmiddle.context import ContextEngineer

# First-time setup: create tables
store = ChatStorage.create(
    "supabase",
    auto_create_tables=True,
    enable_facts=True  # Enable semantic memory tables
)

# In middleware
ContextEngineer(
    model="openai:gpt-4o",
    embedder="openai:text-embedding-3-small",
    backend="supabase",
    backend_kwargs={'enable_facts': True}
)
```

**Context requirements:**
```python
context=StorageContext(
    thread_id="conversation-123",
    user_id="user-456",
    auth_token="jwt-token-from-supabase-auth"  # Required for RLS
)
```

</details>

<details>
<summary><b>Firebase</b> — Real-time NoSQL database</summary>

**Service Account Setup:**
1. Download service account JSON from Firebase Console
2. Set environment variable OR pass path directly

**Option 1: Environment variable**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/firebase-creds.json"
```

**Option 2: Direct path**
```python
ChatSaver(
    backend="firebase",
    credentials_path="./firebase-creds.json"
)
```

**Context requirements:**
```python
context=StorageContext(
    thread_id="conversation-123",
    user_id="user-456",
    auth_token="firebase-id-token"  # From Firebase Auth
)
```

</details>

---

## 📚 Examples

### 🎓 Complete Examples

<details>
<summary><b>1. Simple Chat Bot with Persistence</b></summary>

```python
from langchain.agents import create_agent
from langmiddle.history import ChatSaver, StorageContext

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    context_schema=StorageContext,
    middleware=[ChatSaver(backend="sqlite", db_path="./chatbot.db")]
)

# Conversation 1
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice"}]},
    context=StorageContext(thread_id="thread-1", user_id="alice")
)

# Conversation 2 (same thread - history preserved!)
agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=StorageContext(thread_id="thread-1", user_id="alice")
)
# Response: "Your name is Alice"
```

</details>

<details>
<summary><b>2. Agent with Tools (Clean History)</b></summary>

```python
from langchain.agents import create_agent
from langmiddle.history import ChatSaver, ToolRemover, StorageContext

def search_tool(query: str) -> str:
    return f"Search results for: {query}"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool],
    context_schema=StorageContext,
    middleware=[
        ToolRemover(when="both"),  # Remove tool clutter
        ChatSaver(backend="sqlite", db_path="./agent.db")
    ]
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Search for LangGraph tutorials"}]},
    context=StorageContext(thread_id="thread-1", user_id="user-1")
)
# Only user/assistant messages saved, no tool call noise!
```

</details>

<details>
<summary><b>3. Production Agent with Memory (Supabase)</b></summary>

```python
from langchain.agents import create_agent
from langmiddle.context import ContextEngineer
from langmiddle.storage import ChatStorage
import os

# 1. One-time setup
if os.getenv("INIT_TABLES"):
    store = ChatStorage.create(
        "supabase",
        auto_create_tables=True,
        enable_facts=True
    )
    print("✅ Tables created!")
    exit()

# 2. Create agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    context_schema=StorageContext,
    middleware=[
        ContextEngineer(
            model="openai:gpt-4o",
            embedder="openai:text-embedding-3-small",
            backend="supabase",
            max_tokens_before_summarization=5000,
            extraction_interval=3,
            backend_kwargs={'enable_facts': True}
        )
    ]
)

# 3. Use in your app
def chat(user_id: str, thread_id: str, message: str, jwt_token: str):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        context=StorageContext(
            thread_id=thread_id,
            user_id=user_id,
            auth_token=jwt_token
        )
    )
    return response["messages"][-1]["content"]

# Example usage
chat(
    user_id="user-123",
    thread_id="thread-456",
    message="I prefer vegetarian food and hate spicy dishes",
    jwt_token="eyJ..."
)
# Facts extracted: ["User prefers vegetarian food", "User dislikes spicy dishes"]

chat(
    user_id="user-123",
    thread_id="thread-789",
    message="Recommend a restaurant",
    jwt_token="eyJ..."
)
# Agent uses stored preferences to recommend vegetarian, non-spicy options!
```

</details>

<details>
<summary><b>4. Custom Configuration</b></summary>

```python
from langmiddle.context import (
    ContextEngineer,
    ExtractionConfig,
    SummarizationConfig,
    ContextConfig
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    context_schema=StorageContext,
    middleware=[
        ContextEngineer(
            model="openai:gpt-4o",
            embedder="openai:text-embedding-3-small",
            backend="supabase",

            # Custom extraction settings
            extraction_config=ExtractionConfig(
                interval=5,              # Extract every 5 turns
                max_tokens=2000,         # Or when 2k tokens accumulated
                prompt="<custom prompt>"  # Override extraction prompt
            ),

            # Custom summarization settings
            summarization_config=SummarizationConfig(
                max_tokens=8000,         # Summarize at 8k tokens
                keep_ratio=0.3,          # Keep last 30% of messages
                prefix="## Summary:\n"  # Custom prefix
            ),

            # Custom context injection
            context_config=ContextConfig(
                core_namespaces=[        # Custom always-loaded categories
                    ["user", "profile"],
                    ["user", "preferences"],
                    ["project", "settings"]
                ]
            ),

            backend_kwargs={'enable_facts': True}
        )
    ]
)
```

</details>

---

## 🔍 How It Works

### Message Flow

```
User Input
    ↓
[ToolRemover] ← Cleans tool messages (optional)
    ↓
[ContextEngineer.before_agent] ← Injects facts + summary
    ↓
🤖 LangGraph Agent
    ↓
[ContextEngineer.after_agent] ← Extracts new facts
    ↓
[ChatSaver] ← Persists conversation
    ↓
Response
```

### Fact Lifecycle

```
Conversation → Extraction → Deduplication → Embedding → Storage
                                ↓                          ↓
                          Query & Retrieve         Relevance Scoring
                                ↓                (recency + access + usage)
                          Context Injection              ↓
                          (adaptive detail)      Combined Score
                                ↓                (70% similarity
                               Agent              + 30% relevance)
```

**Phase 3 Relevance Scoring:**
- **Recency** (40%): Newer facts score higher (exponential decay over 365 days)
- **Access Frequency** (30%): Facts used more often get boosted
- **Usage Feedback** (30%): Facts appearing in agent responses are prioritized
- **Adaptive Formatting**: High-relevance facts (≥0.8) get full detail, medium (0.5-0.8) compact, low (0.3-0.5) minimal

---

## 🎨 Architecture Highlights

- **🔌 Modular Design**: Mix and match middleware components
- **🎯 Single Responsibility**: Each middleware does one thing well
- **⚡ Performance**: Embedding caching, batch operations, efficient queries
- **🛡️ Type Safety**: Full Pydantic validation and type hints
- **📊 Observable**: Structured logging with operation IDs and metrics
- **🧪 Testable**: Clean abstractions, dependency injection

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/alpha-xone/langmiddle/issues)
- 💡 **Request features** or improvements
- 🔧 **Submit PRs** for bug fixes or new features
- 📖 **Improve docs** with examples or clarifications
- ⭐ **Star the repo** if LangMiddle helped you!

### Development Setup

```bash
git clone https://github.com/alpha-xone/langmiddle.git
cd langmiddle
pip install -e ".[dev]"
pytest
```

---

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## 🌟 Show Your Support

If LangMiddle saves you time or helps your project, please:
- ⭐ **Star the repo** on GitHub
- 🐦 Share on Twitter/X
- 💬 Tell others in the LangChain community

**Built with ❤️ for the LangGraph ecosystem**
