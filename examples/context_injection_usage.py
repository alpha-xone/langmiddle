"""
Example: Context Injection with ContextEngineer middleware.

This example demonstrates how the ContextEngineer middleware:
1. Injects global context from ALWAYS_LOADED_NAMESPACES before each agent turn
2. Retrieves and injects relevant facts based on conversation context
3. Summarizes earlier conversation and injects as context for long conversations
4. Manages message queue with tagged messages that get refreshed on each turn

The middleware uses special tags in `additional_kwargs` to identify and replace
injected context across conversation turns:
- langmiddle/context: Global user profile and preferences
- langmiddle/facts: Relevant facts from previous conversations
- langmiddle/summary: Summary of earlier parts of current conversation

Requirements:
- LangChain/LangGraph v1
- Supabase project with facts enabled
- Chat model (e.g., OpenAI GPT-4, Anthropic Claude)
- Embedding model (e.g., OpenAI text-embedding-3-small)
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from langmiddle.context import ContextEngineer

# =============================================================================
# Example 1: Basic Context Injection Setup
# =============================================================================

def create_agent_with_context_injection():
    """Create an agent with context injection enabled."""

    # Initialize ContextEngineer middleware
    context_engineer = ContextEngineer(
        model="openai:gpt-4",  # Model for extraction and summarization
        embedder="openai:text-embedding-3-small",  # Embedding model
        backend="supabase",

        # Context injection settings
        enable_context_injection=True,  # Enable before_agent hook
        max_recent_messages=10,  # Keep last 10 messages in context

        # Extraction settings
        max_tokens_before_extraction=2000,  # Trigger extraction after 2000 tokens

        # Backend settings
        backend_kwargs={
            "supabase_url": "YOUR_SUPABASE_URL",
            "supabase_key": "YOUR_SUPABASE_KEY",
            "connection_string": "YOUR_CONNECTION_STRING",
            "enable_facts": True,
        },
    )

    # Add middleware to your agent
    # (Your agent setup here)

    return context_engineer


# =============================================================================
# Example 2: Message Queue Structure
# =============================================================================

"""
How the message queue is restructured by before_agent:

ORIGINAL MESSAGES (Turn 1):
1. HumanMessage("What's my favorite color?")

AFTER BEFORE_AGENT (Turn 1):
1. SystemMessage("## User Profile & Preferences\\n- [user > preferences] Favorite color: blue")
   additional_kwargs: {"langmiddle_tag": "langmiddle/context"}

2. SystemMessage("## Relevant Context\\n- Previous discussion about colors")
   additional_kwargs: {"langmiddle_tag": "langmiddle/facts"}

3. HumanMessage("What's my favorite color?")

AFTER AGENT RESPONDS (Turn 2):
1. SystemMessage(global context) [tagged: langmiddle/context]
2. SystemMessage(relevant facts) [tagged: langmiddle/facts]
3. HumanMessage("What's my favorite color?")
4. AIMessage("Based on your profile, your favorite color is blue!")

USER ASKS AGAIN (Turn 3):
1. SystemMessage(FRESH global context) [tagged: langmiddle/context] ← REPLACED
2. SystemMessage(FRESH relevant facts) [tagged: langmiddle/facts] ← REPLACED
3. HumanMessage("What's my favorite color?")
4. AIMessage("Based on your profile, your favorite color is blue!")
5. HumanMessage("Actually, tell me about my work preferences")

The tagged messages are REPLACED with fresh context on each turn, ensuring
the agent always has up-to-date information.
"""


# =============================================================================
# Example 3: Global Context (ALWAYS_LOADED_NAMESPACES)
# =============================================================================

"""
The middleware automatically loads facts from these namespaces before every turn:

ALWAYS_LOADED_NAMESPACES = [
    ["user", "personal_info"],           # Name, age, location, etc.
    ["user", "professional"],            # Job, industry, skills
    ["user", "preferences", "communication"],  # How they like to communicate
    ["user", "preferences", "formatting"],     # Output format preferences
    ["user", "preferences", "topics"],         # Topics of interest
]

These facts are injected as a SystemMessage with tag "langmiddle/context".

Example output:
```
## User Profile & Preferences
- [user > personal_info] Name is John Doe
- [user > professional] Works as a Software Engineer at Tech Corp
- [user > preferences > communication] Prefers concise, technical responses
- [user > preferences > formatting] Likes code examples with comments
```
"""


# =============================================================================
# Example 4: Relevant Facts Retrieval with Deduplication and Filtering
# =============================================================================

"""
On each turn, the middleware performs a multi-stage process:

**Stage 1: Retrieval**
1. Takes the last 5 messages as context
2. Generates an embedding from their combined content
3. Queries the facts database for relevant memories (threshold: 0.70)
4. Returns top 15 most relevant facts

**Stage 2: Deduplication**
5. Compares with global context facts
6. Removes facts with same ID (exact duplicates)
7. Removes facts with identical content (case-insensitive)
8. Prevents redundant information in context

**Stage 3: LLM Filtering**
9. Uses the model to assess relevance to current conversation
10. Filters out facts that don't add value despite high embedding similarity
11. Returns only facts useful for the current discussion
12. Skips filtering if ≤3 facts (not worth overhead)

**Stage 4: Conditional Injection**
13. Only injects if facts remain after deduplication and filtering
14. Avoids empty context messages

These facts are injected as a SystemMessage with tag "langmiddle/facts".

Example: If user asks "What did we discuss about Python?"

**Retrieved (15 facts at 0.70+ similarity):**
- Previous Python code discussions
- Python libraries mentioned
- Python-related preferences
- Past Python projects discussed
- Other programming discussions (JavaScript, Java)
- General coding preferences

**After Deduplication (10 facts):**
- Removed facts already in global context
- Removed duplicate "prefers Python" fact
- Removed general coding preferences (in global)

**After LLM Filtering (5 facts):**
- [projects > python] Discussed web scraping with BeautifulSoup (confidence: 0.85)
- [learning > python] User is learning FastAPI framework (confidence: 0.92)
- [preferences > python] Prefers type hints in Python code (confidence: 0.88)
- [projects > python] Built CLI tool with Click library (confidence: 0.82)
- [debugging > python] Uses pytest for testing (confidence: 0.90)

Filtered out:
- JavaScript/Java discussions (not relevant to Python question)
- General programming concepts (not specific enough)
- Old Python discussions from months ago (low current relevance)

Final output:
```
## Relevant Context from Previous Conversations
- [projects > python] Discussed building a web scraper with BeautifulSoup (confidence: 0.85)
- [learning > python] User is learning FastAPI framework (confidence: 0.92)
- [preferences > python] Prefers type hints in Python code (confidence: 0.88)
- [projects > python] Built CLI tool with Click library (confidence: 0.82)
- [debugging > python] Uses pytest for testing (confidence: 0.90)
```
"""


# =============================================================================
# Example 5: Conversation Summarization
# =============================================================================

"""
For long conversations (> max_recent_messages), the middleware:
1. Takes all messages EXCEPT the last N (max_recent_messages)
2. Generates a summary using the chat model
3. Injects summary as AIMessage with tag "langmiddle/summary"

This keeps token counts manageable while preserving conversation context.

Example: After 20 messages, with max_recent_messages=10
- Messages 1-10: Summarized
- Messages 11-20: Kept as-is

Summary format:
```
## Previous Conversation Summary
- User asked about building a Python web scraper
- Discussed BeautifulSoup vs Scrapy tradeoffs
- Decided to use BeautifulSoup for simplicity
- Provided example code for parsing HTML tables
- User successfully implemented the scraper
```

This summary is injected AFTER global context and relevant facts, but
BEFORE the recent messages.
"""


# =============================================================================
# Example 6: Complete Agent Flow
# =============================================================================

def complete_agent_example():
    """Complete example showing context injection in action."""

    # Your agent state and graph setup
    class AgentState:
        messages: list

    # Create agent graph
    graph = StateGraph(AgentState)

    # Add your agent node
    # graph.add_node("agent", your_agent_function)

    # Initialize context engineer
    context_engineer = ContextEngineer(
        model="openai:gpt-4",
        embedder="openai:text-embedding-3-small",
        backend="supabase",
        enable_context_injection=True,
        max_recent_messages=10,
        max_tokens_before_extraction=2000,
        backend_kwargs={
            "supabase_url": "YOUR_SUPABASE_URL",
            "supabase_key": "YOUR_SUPABASE_KEY",
            "connection_string": "YOUR_CONNECTION_STRING",
            "enable_facts": True,
        },
    )

    # Compile with middleware
    # app = graph.compile(middlewares=[context_engineer])

    # Create runtime with user credentials
    runtime = Runtime(context={
        "user_id": "user_123",
        "auth_token": "jwt_token_here"
    })

    # Invoke agent
    # result = app.invoke(
    #     {"messages": [HumanMessage(content="What's my favorite color?")]},
    #     runtime=runtime
    # )

    print("Agent executed with context injection")


# =============================================================================
# Example 7: Debugging and Monitoring
# =============================================================================

def monitoring_example():
    """Monitor context injection behavior."""

    context_engineer = ContextEngineer(
        model="openai:gpt-4",
        embedder="openai:text-embedding-3-small",
        backend="supabase",
        enable_context_injection=True,
        backend_kwargs={
            "supabase_url": "YOUR_SUPABASE_URL",
            "supabase_key": "YOUR_SUPABASE_KEY",
            "connection_string": "YOUR_CONNECTION_STRING",
            "enable_facts": True,
        },
    )

    # Check embeddings cache stats
    cache_stats = context_engineer.get_cache_stats()
    print(f"Cache size: {cache_stats['size']}")
    print(f"Sample keys: {cache_stats['sample_keys']}")

    # Clear cache if needed
    context_engineer.clear_embeddings_cache()

    # Monitor logs
    # The middleware logs detailed information about:
    # - Number of global facts loaded
    # - Number of relevant facts retrieved
    # - Number of recent messages kept
    # - Whether conversation summary was generated

    """
    Example log output:
    [before_agent] Injected global context
    [before_agent] Injected relevant facts
    [before_agent] Injected conversation summary
    [before_agent] Context injection complete: 5 global facts, 12 relevant facts, 10 recent messages
    """


# =============================================================================
# Example 8: Customizing Context Injection
# =============================================================================

def custom_context_example():
    """Customize context injection behavior."""

    # Option 1: Adjust recent message window
    context_engineer_short = ContextEngineer(
        model="openai:gpt-4",
        embedder="openai:text-embedding-3-small",
        backend="supabase",
        max_recent_messages=5,  # Keep fewer messages (save tokens)
        enable_context_injection=True,
        backend_kwargs={"supabase_url": "..."},
    )

    # Option 2: Disable context injection (extraction only)
    context_engineer_extract_only = ContextEngineer(
        model="openai:gpt-4",
        embedder="openai:text-embedding-3-small",
        backend="supabase",
        enable_context_injection=False,  # Only extract, don't inject
        backend_kwargs={"supabase_url": "..."},
    )

    # Option 3: Aggressive summarization
    context_engineer_aggressive = ContextEngineer(
        model="openai:gpt-4",
        embedder="openai:text-embedding-3-small",
        backend="supabase",
        max_recent_messages=3,  # Very short window
        max_tokens_before_extraction=1000,  # Frequent extraction
        enable_context_injection=True,
        backend_kwargs={"supabase_url": "..."},
    )


# =============================================================================
# Example 9: Deduplication and Filtering in Action
# =============================================================================

"""
Real-world example showing how deduplication and filtering improve context quality:

Scenario: User asks "How should I structure my Python API?"

**Global Context (always loaded):**
- [user > preferences > communication] Prefers concise, practical responses
- [user > preferences > formatting] Likes code examples with comments
- [user > professional] Works as Backend Developer
- [user > preferences > topics] Interested in Python, APIs, DevOps

**Retrieved Facts (15 facts, 0.70+ similarity):**
1. [projects > python] Built REST API with Flask last year
2. [user > preferences > communication] Prefers concise, practical responses  ← DUPLICATE
3. [deployment > docker] Uses Docker for containerization
4. [learning > python] Learning FastAPI framework
5. [projects > javascript] Built React frontend with REST client
6. [user > professional] Works as Backend Developer  ← DUPLICATE
7. [deployment > aws] Deployed services to AWS Lambda
8. [learning > databases] Exploring PostgreSQL optimization
9. [projects > python] Built CLI tool with Click
10. [preferences > code_style] Prefers functional programming
11. [testing > python] Uses pytest and pytest-cov
12. [deployment > ci_cd] Uses GitHub Actions for CI/CD
13. [learning > architecture] Reading about microservices patterns
14. [projects > old] Built PHP website 5 years ago
15. [user > preferences > formatting] Likes code examples  ← DUPLICATE

**After Deduplication (12 facts):**
Removed 3 duplicates that are already in global context:
- Fact 2: preferences > communication
- Fact 6: professional info
- Fact 15: preferences > formatting

**LLM Filtering Decision:**
Analyzing against question: "How should I structure my Python API?"

Relevant (keep):
✓ 1. Flask REST API experience (directly relevant)
✓ 3. Docker containerization (API deployment)
✓ 4. Learning FastAPI (directly relevant)
✓ 7. AWS Lambda deployment (API hosting option)
✓ 8. PostgreSQL optimization (API data layer)
✓ 11. pytest testing (API testing)
✓ 13. Microservices patterns (API architecture)

Not relevant (filter out):
✗ 5. React frontend (not about API structure)
✗ 9. CLI tool (different domain)
✗ 10. Functional programming (too general)
✗ 12. GitHub Actions (not about structure)
✗ 14. PHP website (outdated, wrong stack)

**Final Injected Context (7 facts):**
```
## Relevant Context from Previous Conversations
- [projects > python] Built REST API with Flask last year (confidence: 0.88)
- [deployment > docker] Uses Docker for containerization (confidence: 0.85)
- [learning > python] Learning FastAPI framework (confidence: 0.92)
- [deployment > aws] Deployed services to AWS Lambda (confidence: 0.83)
- [learning > databases] Exploring PostgreSQL optimization (confidence: 0.80)
- [testing > python] Uses pytest and pytest-cov (confidence: 0.87)
- [learning > architecture] Reading about microservices patterns (confidence: 0.81)
```

**Benefits:**
- Started with 15 retrieved + 4 global = 19 total facts
- Ended with 4 global + 7 relevant = 11 facts (42% reduction)
- All 11 facts are highly relevant and non-redundant
- Saved ~200 tokens while improving quality
"""


# =============================================================================
# Example 10: Understanding Message Tags
# =============================================================================

"""
How to identify tagged messages in your code:

```python
for msg in state["messages"]:
    additional_kwargs = getattr(msg, "additional_kwargs", {})
    tag = additional_kwargs.get("langmiddle_tag")

    if tag == "langmiddle/context":
        print("This is global context (user profile)")
    elif tag == "langmiddle/facts":
        print("This is relevant facts from history")
    elif tag == "langmiddle/summary":
        print("This is conversation summary")
    else:
        print("This is a regular message")
```

Tags are used internally by the middleware to:
1. Identify which messages to replace on subsequent turns
2. Maintain consistent positioning in the message queue
3. Ensure fresh context without duplication
"""


# =============================================================================
# Best Practices
# =============================================================================

"""
1. Token Management:
   - Use max_recent_messages to control context window size
   - Enable summarization for long conversations
   - Monitor token usage in production
   - Deduplication and filtering automatically reduce token usage

2. Cache Management:
   - Embeddings are cached automatically for efficiency
   - Clear cache periodically: context_engineer.clear_embeddings_cache()
   - Check cache stats: context_engineer.get_cache_stats()

3. Context Quality:
   - Ensure ALWAYS_LOADED_NAMESPACES facts are well-organized
   - Use hierarchical namespaces for better filtering
   - Keep fact content concise and informative
   - LLM filtering ensures only relevant facts are injected
   - Deduplication prevents redundant information

4. Performance:
   - Context injection adds latency:
     * ~150-300ms without filtering
     * ~450-800ms with filtering (only if >3 relevant facts)
   - Deduplication is very fast (~1ms)
   - LLM filtering only runs when beneficial (>3 facts)
   - Consider enabling only for user-facing interactions
   - Use enable_context_injection=False for internal tools

5. Debugging:
   - Enable verbose logging to see what's injected
   - Inspect message additional_kwargs to verify tags
   - Monitor fact retrieval, deduplication, and filtering counts in logs
   - Check for "X retrieved, Y after deduplication, Z after filtering" messages

6. Security:
   - Always provide proper credentials (user_id + auth_token)
   - Respect RLS policies in your database
   - Don't inject facts from other users
   - Deduplication uses ID and content, not embeddings (more secure)

7. Quality Assurance:
   - Review LLM filtering decisions periodically
   - Monitor how many facts are filtered out
   - Adjust match_threshold (0.70) if filtering too aggressively
   - Check that deduplication catches all duplicates
"""


if __name__ == "__main__":
    print("🧠 Context Injection with ContextEngineer")
    print("=" * 50)
    print("\nKey Features:")
    print("  - Automatic global context injection (user profile)")
    print("  - Relevant facts retrieval based on conversation")
    print("  - Smart deduplication (removes duplicates from global context)")
    print("  - LLM-based filtering (ensures relevance to current discussion)")
    print("  - Conversation summarization for long interactions")
    print("  - Tagged messages for efficient context updates")
    print("\nMessage Tags:")
    print("  - langmiddle/context: Global user profile")
    print("  - langmiddle/facts: Relevant historical facts (deduplicated & filtered)")
    print("  - langmiddle/summary: Conversation summary")
    print("\nQuality Improvements:")
    print("  - Deduplication: Removes redundant facts (~1ms)")
    print("  - LLM Filtering: Keeps only relevant facts (~300-500ms when >3 facts)")
    print("  - Conditional Injection: Only injects when facts exist")
    print("\nSee code comments for detailed examples and usage patterns.")
