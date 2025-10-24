"""
Example: Multi-Backend Storage Usage

This example demonstrates how to use the langmiddle storage system
with different backends: Supabase, SQLite, and Firebase.
"""

from langchain_core.messages import AIMessage, HumanMessage

from langmiddle.storage import ChatStorage
from langmiddle.utils.storage import save_chat_history

# Create some sample messages
messages = [
    HumanMessage(content="Hello, can you help me with my project?", id="msg_1"),
    AIMessage(
        content="Of course! I'd be happy to help. What kind of project are you working on?",
        id="msg_2",
    ),
    HumanMessage(
        content="I'm building a chat application with message persistence.", id="msg_3"
    ),
    AIMessage(
        content="Great! You can use different storage backends based on your needs.",
        id="msg_4",
    ),
]

print("=== LangMiddle Multi-Backend Storage Examples ===\n")

# Example 1: SQLite (Local File Storage)
print("1. SQLite Local File Storage:")
print("   Best for: Development, local applications, offline usage")

result = save_chat_history(
    thread_id="example_thread_sqlite",
    auth_token=None,
    messages=messages,
    user_id="user_123",
    backend_type="sqlite",
    db_path="./chat_history.db",  # Local file
)

print(f"   ✓ Saved {result['saved_count']} messages to local SQLite file")
print(f"   ✓ Success: {result['success']}")

# Example 2: SQLite (In-Memory)
print("\n2. SQLite In-Memory Storage:")
print("   Best for: Testing, temporary data, demos")

result = save_chat_history(
    thread_id="example_thread_memory",
    auth_token=None,
    messages=messages[:2],  # Fewer messages for demo
    user_id="user_456",
    backend_type="sqlite",
    db_path=":memory:",  # In-memory database
)

print(f"   ✓ Saved {result['saved_count']} messages to memory")
print(f"   ✓ Success: {result['success']}")

# Example 3: Direct ChatStorage Interface
print("\n3. Direct ChatStorage Interface:")
print("   Best for: Advanced usage, custom integrations")

storage = ChatStorage.create("sqlite", db_path=":memory:")
result = storage.save_chat_history(
    thread_id="direct_example",
    credentials={"user_id": "user_789"},
    messages=messages[2:],  # Last two messages
)

print(f"   ✓ Saved {result['saved_count']} messages via direct interface")
print(f"   ✓ Success: {result['success']}")

# Example 4: Production Examples (commented out - require actual credentials)
print("\n4. Production Backend Examples:")
print("   (Uncomment and configure credentials to use)")

print(
    """
# Supabase (Cloud PostgreSQL with RLS)
# Best for: Production web apps, multi-user systems, real-time features

result = save_chat_history(
    thread_id="prod_thread",
    auth_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",  # Your JWT token
    messages=messages,
    user_id="user_123",  # Optional, extracted from auth_token
    backend_type="supabase"
)

# Firebase (Cloud Firestore)
# Best for: Google ecosystem, mobile apps, real-time sync

result = save_chat_history(
    thread_id="firebase_thread",
    auth_token="firebase_id_token...",  # Firebase ID token
    messages=messages,
    user_id="user_456",
    backend_type="firebase",
    credentials_path="path/to/firebase-credentials.json"
)
"""
)

# Example 5: Middleware Integration
print("\n5. Middleware Integration:")
print("   (Example of how ChatSaver uses the storage system)")

print(
    """
# In your LangGraph application:
from langmiddle import ChatSaver

# Create middleware with backend selection
history_middleware = ChatSaver(
    backend="sqlite",  # or "supabase", "firebase"
    db_path="./chat.db"  # SQLite-specific config
)

# Use in your graph
graph = graph.with_middleware([history_middleware])
"""
)

print("\n=== Benefits of Multi-Backend Approach ===")
print("✓ Same API across all storage backends")
print("✓ Easy switching between development and production")
print("✓ Backward compatibility with existing code")
print("✓ Flexible deployment options")
print("✓ Database-agnostic application design")

print("\n=== Backend Comparison ===")
print("SQLite:    Local, embedded, great for development")
print("Supabase:  Cloud PostgreSQL, RLS, real-time, multi-user")
print("Firebase:  Cloud Firestore, Google ecosystem, mobile-friendly")

print("\nChoose the backend that best fits your deployment needs!")
