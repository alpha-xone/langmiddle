"""Example: Using the new optimal context API.

This example demonstrates the new protocol-based, stateless architecture
with pure dependency injection. This is the recommended approach.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langmiddle.context import ContextConfig, create_middleware
from langmiddle.storage import SupabaseBackend

# Example setup
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"


def basic_usage():
    """Most common use case: default config with minimal setup."""

    # Step 1: Create storage backend
    storage = SupabaseBackend(url=SUPABASE_URL, key=SUPABASE_KEY)

    # Step 2: Create embedder and model
    embedder = OpenAIEmbeddings()
    model = ChatOpenAI(model="gpt-4", temperature=0)

    # Step 3: Create middleware with defaults
    middleware = create_middleware(
        storage=storage,
        embedder=embedder,
        model=model,
    )

    # Step 4: Use it
    conversation = [
        HumanMessage(content="My name is Alice"),
        AIMessage(content="Nice to meet you, Alice!"),
        HumanMessage(content="What's my name?"),
    ]

    result = middleware.inject_context(
        messages=conversation,
        user_id="user123",
        credentials={"supabase_auth": "your-token"},
    )

    print(f"Input: {len(conversation)} messages")
    print(f"Output: {len(result)} messages (with context injected)")

    return result


def custom_config():
    """Advanced use case: custom configuration."""

    storage = SupabaseBackend(url=SUPABASE_URL, key=SUPABASE_KEY)
    embedder = OpenAIEmbeddings()
    model = ChatOpenAI(model="gpt-4")

    # Create custom configuration
    config = ContextConfig.create(
        # Keep more recent messages
        max_recent_messages=15,
        # Adjust retrieval settings
        match_threshold=0.75,
        match_count=10,
        context_window=8,
        # Disable filtering for this use case
        enable_filtering=False,
        # Adjust summarization
        summary_threshold=20,
        min_messages=5,
    )

    # Create middleware with custom config
    middleware = create_middleware(
        storage=storage,
        embedder=embedder,
        model=model,
        config=config,
    )

    conversation = [
        HumanMessage(content="I love Python programming"),
        AIMessage(content="Python is great!"),
        HumanMessage(content="What do I like?"),
    ]

    result = middleware.inject_context(
        messages=conversation,
        user_id="user456",
        credentials={"supabase_auth": "your-token"},
    )

    return result


def minimal_config():
    """Minimal use case: only override what you need."""

    storage = SupabaseBackend(url=SUPABASE_URL, key=SUPABASE_KEY)
    embedder = OpenAIEmbeddings()
    model = ChatOpenAI()

    # Only override the settings you care about
    config = ContextConfig.create(
        max_recent_messages=5,  # Just this one setting
    )

    middleware = create_middleware(
        storage=storage,
        embedder=embedder,
        model=model,
        config=config,
    )

    conversation = [HumanMessage(content="Hello!")]

    return middleware.inject_context(
        messages=conversation,
        user_id="user789",
        credentials={"supabase_auth": "your-token"},
    )


def with_caching():
    """Use embeddings caching for better performance."""

    storage = SupabaseBackend(url=SUPABASE_URL, key=SUPABASE_KEY)
    embedder = OpenAIEmbeddings()
    model = ChatOpenAI()

    middleware = create_middleware(
        storage=storage,
        embedder=embedder,
        model=model,
    )

    conversation = [
        HumanMessage(content="I work in San Francisco"),
        AIMessage(content="Great city!"),
        HumanMessage(content="Where do I work?"),
    ]

    # Create cache for embeddings (reuse across calls)
    embeddings_cache: dict = {}

    result = middleware.inject_context(
        messages=conversation,
        user_id="user999",
        credentials={"supabase_auth": "your-token"},
        cache=embeddings_cache,
    )

    print(f"Cache now contains: {len(embeddings_cache)} entries")

    return result


def inspect_config():
    """Check the current configuration."""

    storage = SupabaseBackend(url=SUPABASE_URL, key=SUPABASE_KEY)
    embedder = OpenAIEmbeddings()
    model = ChatOpenAI()

    middleware = create_middleware(
        storage=storage,
        embedder=embedder,
        model=model,
    )

    # Access immutable config
    config = middleware.config

    print(f"Max recent messages: {config.max_recent_messages}")
    print(f"Match threshold: {config.retrieval.match_threshold}")
    print(f"Match count: {config.retrieval.match_count}")
    print(f"Context window: {config.retrieval.context_window}")
    print(f"Deduplication enabled: {config.processing.enable_deduplication}")
    print(f"Filtering enabled: {config.processing.enable_filtering}")
    print(f"Filter threshold: {config.processing.filter_threshold}")
    print(f"Summary threshold: {config.summarization.summary_threshold}")


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMAL CONTEXT API EXAMPLES")
    print("=" * 60)

    print("\n1. Basic Usage (default config)")
    print("-" * 60)
    basic_usage()

    print("\n2. Custom Configuration")
    print("-" * 60)
    custom_config()

    print("\n3. Minimal Configuration")
    print("-" * 60)
    minimal_config()

    print("\n4. With Embeddings Caching")
    print("-" * 60)
    with_caching()

    print("\n5. Inspect Configuration")
    print("-" * 60)
    inspect_config()
