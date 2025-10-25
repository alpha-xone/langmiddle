# Memory Extraction Middleware

Automatically extract and store memories/insights from conversations using LLM-based analysis.

## Overview

The `MemoryExtractor` middleware analyzes conversation messages after each agent run to identify and extract key information that should be stored for long-term memory. It uses LLM-based extraction with structured output to generate memories in a format compatible with LangGraph Store implementations (PostgresStore, InMemoryStore, etc.).

## Features

- 🧠 **LLM-Powered Extraction**: Uses advanced language models to identify important information
- 📦 **Store-Compatible Format**: Output works with PostgresStore, InMemoryStore, and other Store implementations
- 🎯 **Customizable Prompts**: Define your own extraction criteria via custom prompts
- 🏷️ **Namespace Organization**: Hierarchical namespace structure for organized memory storage
- ⏱️ **TTL Support**: Optional time-to-live for automatic memory expiration
- 🔍 **Smart Filtering**: Automatically filters out tool messages to focus on conversation content

## Installation

```bash
pip install langmiddle
```

## Quick Start

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langmiddle.memory import MemoryExtractor

# Initialize the extractor
extractor = MemoryExtractor(
    model=ChatOpenAI(model="gpt-4"),
    namespace_prefix=["user", "123"]
)

# Use in your LangGraph workflow
# The middleware will automatically extract memories after each agent run
```

### Custom Prompt

```python
from langchain_core.prompts import ChatPromptTemplate

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract user preferences from the conversation.
Focus on:
- Personal information
- Preferences and interests
- Important facts
"""),
    ("user", "Conversation:\n{messages}")
])

extractor = MemoryExtractor(
    model=ChatOpenAI(model="gpt-4"),
    namespace_prefix=["user", "preferences"],
    prompt_template=custom_prompt
)
```

## Memory Format

Extracted memories follow the PostgresStore format:

```python
{
    "namespace": ("user", "123"),  # Hierarchical tuple
    "key": "favorite_language",     # Unique identifier
    "value": {                       # Structured dictionary
        "language": "Python",
        "reason": "Loves the simplicity",
        "confidence": "high"
    },
    "ttl": 10080.0  # Optional TTL in minutes (7 days)
}
```

## Integration Examples

### With InMemoryStore

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
extractor = MemoryExtractor(
    model=ChatOpenAI(model="gpt-4"),
    namespace_prefix=["session", "abc123"]
)

# After extraction, store memories
result = extractor.after_agent(state, runtime)
if result and "memories" in result:
    for memory in result["memories"]:
        store.put(
            namespace=memory["namespace"],
            key=memory["key"],
            value=memory["value"],
            ttl=memory.get("ttl")
        )
```

### With PostgresStore

```python
from langgraph.store.postgres import AsyncPostgresStore

async with AsyncPostgresStore.from_conn_string(conn_string) as store:
    await store.setup()

    extractor = MemoryExtractor(
        model=ChatOpenAI(model="gpt-4"),
        namespace_prefix=["user", "456"]
    )

    # Extract and store
    result = extractor.after_agent(state, runtime)
    if result and "memories" in result:
        for memory in result["memories"]:
            await store.aput(
                namespace=memory["namespace"],
                key=memory["key"],
                value=memory["value"],
                ttl=memory.get("ttl")
            )
```

## Configuration Options

### Model

The `model` parameter accepts either:
- A model name string: `"gpt-4"`, `"gpt-3.5-turbo"`
- A `BaseChatModel` instance: `ChatOpenAI(model="gpt-4")`

### Namespace Prefix

The `namespace_prefix` defines the hierarchical path for organizing memories:

```python
# User-specific memories
namespace_prefix=["user", "user123"]

# Session-specific memories
namespace_prefix=["session", "session456"]

# Multi-level organization
namespace_prefix=["org", "team", "project"]
```

### Prompt Template

Custom prompts allow you to control what gets extracted:

```python
# Focus on technical information
technical_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract technical skills and tools mentioned."),
    ("user", "{messages}")
])

# Focus on personal preferences
preference_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract user preferences and likes/dislikes."),
    ("user", "{messages}")
])
```

## Extraction Process

1. **Message Filtering**: Tool messages are automatically filtered out
2. **Formatting**: Messages are formatted into a readable conversation
3. **LLM Invocation**: The model extracts structured memories using the prompt
4. **Namespace Assignment**: Default namespace prefix is applied if not set
5. **Return**: Memories are returned in Store-compatible format

## Data Models

### MemoryItem

```python
class MemoryItem(BaseModel):
    namespace: list[str]        # Hierarchical path
    key: str                     # Unique identifier
    value: dict[str, Any]        # Memory content
    ttl: float | None = None     # Optional expiration
```

### MemoriesExtraction

```python
class MemoriesExtraction(BaseModel):
    memories: list[MemoryItem] = []  # Extracted memories
```

## Best Practices

1. **Namespace Design**: Use consistent namespace structures across your application
   ```python
   ["user", user_id]
   ["session", session_id]
   ["team", team_id, "member", user_id]
   ```

2. **Key Naming**: Use descriptive, consistent key names
   ```python
   "preferred_language"  # Good
   "pref_lang"          # Avoid abbreviations
   ```

3. **Value Structure**: Store structured data with clear field names
   ```python
   {
       "language": "Python",
       "proficiency": "expert",
       "years_experience": 5
   }
   ```

4. **TTL Strategy**: Set appropriate TTLs based on data volatility
   ```python
   ttl=1440.0      # 24 hours for session data
   ttl=10080.0     # 7 days for user preferences
   ttl=43200.0     # 30 days for long-term facts
   ttl=None        # Never expire for permanent data
   ```

## Logging

The middleware uses structured logging to track extraction activity:

```python
logger.info(f"Extracted {len(result.memories)} memories from conversation")
logger.debug("No non-tool messages found, skipping extraction")
logger.error(f"Error extracting memories: {e}")
```

## Error Handling

The middleware gracefully handles errors and returns `None` on failure:

```python
try:
    result = extractor.after_agent(state, runtime)
except Exception as e:
    logger.error(f"Error in memory extraction: {e}")
    return None
```

## Performance Considerations

- **Message Filtering**: Only non-tool messages are processed to reduce token usage
- **Batch Processing**: Consider batching extractions for high-volume applications
- **Model Selection**: Balance accuracy vs. speed by choosing appropriate models
- **Caching**: Cache frequently extracted patterns to reduce API calls

## Examples

See `examples/memory_usage.py` for complete working examples including:
- Basic memory extraction
- Custom prompts and configuration
- PostgreSQL integration
- LangGraph workflow integration

## Comparison with History Middleware

| Feature | HistorySaver | MemoryExtractor |
|---------|-------------|-----------------|
| Purpose | Save raw messages | Extract insights |
| When | After each message | After agent run |
| Output | Raw chat history | Structured memories |
| Storage | SQLite/Supabase/Firebase | LangGraph Stores |
| Processing | Direct storage | LLM analysis |

## Related Modules

- `langmiddle.history` - Chat history persistence
- `langgraph.store` - Memory storage implementations
- `langchain.agents.middleware` - Middleware framework

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
