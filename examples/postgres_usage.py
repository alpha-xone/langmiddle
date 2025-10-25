"""
PostgreSQL Backend Usage Example

This example demonstrates how to use the PostgreSQL storage backend
with langmiddle.
"""

from langmiddle import ChatStorage
from langchain_core.messages import HumanMessage, AIMessage


def basic_usage():
    """Basic PostgreSQL backend usage with connection string."""

    # Create storage with PostgreSQL backend
    storage = ChatStorage.create(
        "postgres",
        connection_string="postgresql://user:password@localhost:5432/chatdb",
        auto_create_tables=True
    )

    # Create sample messages
    messages = [
        HumanMessage(content="Hello, how are you?", id="msg-1"),
        AIMessage(content="I'm doing well, thank you!", id="msg-2"),
    ]

    # Save messages
    result = storage.save_chat_history(
        thread_id="thread-123",
        credentials={"user_id": "user-456"},
        messages=messages
    )

    print(f"Saved {result['saved_count']} messages")
    print(f"Errors: {result['errors']}")


def env_usage():
    """PostgreSQL backend using environment variables."""

    # Create .env file with:
    # POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/chatdb

    storage = ChatStorage.create("postgres", auto_create_tables=True)

    messages = [
        HumanMessage(content="What's the weather?", id="msg-3"),
        AIMessage(content="It's sunny today!", id="msg-4"),
    ]

    result = storage.save_chat_history(
        thread_id="thread-789",
        credentials={"user_id": "user-456"},
        messages=messages
    )

    print(f"Success: {result['success']}")


def incremental_save():
    """Demonstrate incremental message saving to avoid duplicates."""

    storage = ChatStorage.create(
        "postgres",
        connection_string="postgresql://user:password@localhost:5432/chatdb"
    )

    thread_id = "thread-abc"
    credentials = {"user_id": "user-456"}

    # First save
    messages_batch_1 = [
        HumanMessage(content="Tell me a joke", id="msg-5"),
        AIMessage(content="Why did the chicken cross the road?", id="msg-6"),
    ]

    result1 = storage.save_chat_history(
        thread_id=thread_id,
        credentials=credentials,
        messages=messages_batch_1
    )

    print(f"First batch: saved {result1['saved_count']} messages")
    saved_msg_ids = result1['saved_msg_ids']

    # Second save (with some overlapping messages)
    messages_batch_2 = [
        AIMessage(content="Why did the chicken cross the road?", id="msg-6"),  # duplicate
        HumanMessage(content="I don't know, why?", id="msg-7"),  # new
        AIMessage(content="To get to the other side!", id="msg-8"),  # new
    ]

    result2 = storage.save_chat_history(
        thread_id=thread_id,
        credentials=credentials,
        messages=messages_batch_2,
        saved_msg_ids=saved_msg_ids  # Pass previous saved IDs to avoid duplicates
    )

    print(f"Second batch: saved {result2['saved_count']} new messages")
    print(f"Skipped {result2['skipped_count']} duplicate messages")


def comparison_with_supabase():
    """Compare PostgreSQL and Supabase backends side by side."""

    # PostgreSQL - Direct database connection
    postgres_storage = ChatStorage.create(
        "postgres",
        connection_string="postgresql://user:password@localhost:5432/chatdb"
    )

    # Supabase - Uses Supabase's REST API + PostgreSQL
    supabase_storage = ChatStorage.create(
        "supabase",
        supabase_url="https://your-project.supabase.co",
        supabase_key="your-anon-key"
    )

    messages = [
        HumanMessage(content="Test message", id="msg-9"),
    ]

    # Both use the same interface!
    postgres_result = postgres_storage.save_chat_history(
        thread_id="thread-pg",
        credentials={"user_id": "user-123"},
        messages=messages
    )

    supabase_result = supabase_storage.save_chat_history(
        thread_id="thread-sb",
        credentials={"user_id": "user-123"},
        messages=messages
    )

    print("PostgreSQL:", postgres_result['success'])
    print("Supabase:", supabase_result['success'])


if __name__ == "__main__":
    print("=== Basic Usage ===")
    # basic_usage()

    print("\n=== Environment Variables Usage ===")
    # env_usage()

    print("\n=== Incremental Save ===")
    # incremental_save()

    print("\n=== PostgreSQL vs Supabase Comparison ===")
    # comparison_with_supabase()

    print("\nUncomment the function calls to run examples!")
