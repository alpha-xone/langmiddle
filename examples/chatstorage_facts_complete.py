"""
Complete example demonstrating the ChatStorage facts API.

This script shows how to use the unified ChatStorage interface for facts management:
1. Insert facts with embeddings
2. Query facts by similarity
3. Update and delete facts
4. Track processed messages (batch operations)

Unlike facts_usage.py which uses the backend directly, this example uses the
ChatStorage unified interface which handles authentication automatically.
"""

import os
from typing import List

from dotenv import load_dotenv

from langmiddle.storage import ChatStorage

# Load environment variables
load_dotenv()


def get_mock_embedding(text: str, dimension: int = 1536) -> List[float]:
    """
    Generate a mock embedding for demonstration.

    In production, replace this with a real embedding model:
    - OpenAI: langchain_openai.OpenAIEmbeddings
    - Sentence Transformers: langchain_huggingface.HuggingFaceEmbeddings
    - Cohere: langchain_cohere.CohereEmbeddings
    """
    import hashlib
    import math

    # Generate deterministic embedding from text hash
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()

    embedding = []
    for i in range(dimension):
        byte_idx = i % len(hash_bytes)
        # Use sine wave pattern with hash as seed for variety
        value = math.sin(hash_bytes[byte_idx] + i * 0.1) * 0.8
        embedding.append(value)

    # Normalize to unit vector
    magnitude = math.sqrt(sum(x * x for x in embedding))
    return [x / magnitude for x in embedding]


def main():
    print("=" * 80)
    print("ChatStorage Facts Management - Complete Example")
    print("=" * 80)
    print()

    # Create unified storage interface with facts enabled
    print("Initializing ChatStorage with Supabase backend...")
    storage = ChatStorage.create(
        "supabase",
        auto_create_tables=True,
        enable_facts=True,  # Enable facts tables
    )
    print("✓ Storage initialized\n")

    # Credentials for authentication
    credentials = {
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_key": os.getenv("SUPABASE_ANON_KEY"),
    }

    # ==========================================================================
    # Example 1: Insert Facts with Embeddings
    # ==========================================================================
    print("=" * 80)
    print("Example 1: Inserting Facts with Embeddings")
    print("=" * 80)

    facts_to_insert = [
        {
            "content": "The user prefers dark mode for all applications",
            "namespace": ["user", "preferences", "ui", "theme"],
            "language": "en",
            "intensity": 0.9,
            "confidence": 0.95,
        },
        {
            "content": "The user is allergic to peanuts and tree nuts",
            "namespace": ["user", "health", "allergies"],
            "language": "en",
            "intensity": 1.0,
            "confidence": 1.0,
        },
        {
            "content": "The user's favorite programming language is Python",
            "namespace": ["user", "preferences", "development", "languages"],
            "language": "en",
            "intensity": 0.85,
            "confidence": 0.9,
        },
        {
            "content": "The user works as a machine learning engineer",
            "namespace": ["user", "professional", "role"],
            "language": "en",
            "intensity": 0.95,
            "confidence": 1.0,
        },
    ]

    # Generate embeddings (using mock for demo - use real embeddings in production)
    model_dimension = 1536  # OpenAI text-embedding-3-small dimension
    embeddings = [
        get_mock_embedding(fact["content"], model_dimension)
        for fact in facts_to_insert
    ]

    # Ensure embedding table exists for this dimension
    print(f"Creating embedding table for dimension {model_dimension}...")
    table_created = storage.get_or_create_embedding_table(credentials, model_dimension)
    if table_created:
        print("✓ Embedding table ready\n")
    else:
        print("✗ Failed to create embedding table\n")
        return

    # Insert facts with embeddings
    print(f"Inserting {len(facts_to_insert)} facts...")
    insert_result = storage.insert_facts(
        credentials=credentials,
        facts=facts_to_insert,
        embeddings=embeddings,
        model_dimension=model_dimension,
    )

    if insert_result["success"]:
        print(f"✓ Successfully inserted {insert_result['inserted_count']} facts")
        print(f"  Fact IDs: {insert_result['fact_ids']}")
    else:
        print(f"✗ Insert failed: {insert_result['errors']}")
        return

    fact_ids = insert_result["fact_ids"]
    print()

    # ==========================================================================
    # Example 2: Query Facts by Similarity
    # ==========================================================================
    print("=" * 80)
    print("Example 2: Querying Facts by Similarity")
    print("=" * 80)

    queries = [
        "What programming languages does the user like?",
        "What are the user's dietary restrictions?",
        "What does the user do for work?",
    ]

    for query_text in queries:
        print(f"\nQuery: '{query_text}'")
        print("-" * 80)

        query_embedding = get_mock_embedding(query_text, model_dimension)

        results = storage.query_facts(
            credentials=credentials,
            query_embedding=query_embedding,
            model_dimension=model_dimension,
            match_threshold=0.3,  # Lower threshold for mock embeddings
            match_count=3,
        )

        if results:
            print(f"Found {len(results)} relevant facts:")
            for i, fact in enumerate(results, 1):
                print(f"\n  {i}. {fact['content']}")
                print(f"     Namespace: {' > '.join(fact['namespace'])}")
                print(f"     Similarity: {fact.get('similarity', 0):.3f}")
                print(f"     Confidence: {fact.get('confidence', 0):.2f}")
        else:
            print("  No matching facts found")

    print()

    # ==========================================================================
    # Example 3: Query with Namespace Filtering
    # ==========================================================================
    print("=" * 80)
    print("Example 3: Query with Namespace Filtering")
    print("=" * 80)

    query_text = "Tell me about the user"
    query_embedding = get_mock_embedding(query_text, model_dimension)

    # Filter to only preferences
    print(f"\nQuery: '{query_text}'")
    print("Filter: namespace contains ['user', 'preferences']")
    print("-" * 80)

    results = storage.query_facts(
        credentials=credentials,
        query_embedding=query_embedding,
        model_dimension=model_dimension,
        match_threshold=0.3,
        match_count=5,
        filter_namespaces=[["user", "preferences"]],
    )

    if results:
        print(f"Found {len(results)} facts in preferences namespace:")
        for i, fact in enumerate(results, 1):
            print(f"\n  {i}. {fact['content']}")
            print(f"     Namespace: {' > '.join(fact['namespace'])}")
    else:
        print("  No matching facts found")

    print()

    # ==========================================================================
    # Example 4: Get, Update, and Delete Facts
    # ==========================================================================
    print("=" * 80)
    print("Example 4: Get, Update, and Delete Facts")
    print("=" * 80)

    if fact_ids:
        fact_id = fact_ids[0]

        # Get fact by ID
        print(f"\n1. Getting fact by ID: {fact_id}")
        print("-" * 80)
        fact = storage.get_fact_by_id(credentials=credentials, fact_id=fact_id)

        if fact:
            print(f"✓ Retrieved: {fact['content']}")
            print(f"  Confidence: {fact.get('confidence', 'N/A')}")
            print(f"  Intensity: {fact.get('intensity', 'N/A')}")

            # Update the fact
            print("\n2. Updating fact confidence and intensity...")
            print("-" * 80)
            updated = storage.update_fact(
                credentials=credentials,
                fact_id=fact_id,
                updates={"confidence": 1.0, "intensity": 0.95},
            )

            if updated:
                print("✓ Fact updated successfully")

                # Verify update
                updated_fact = storage.get_fact_by_id(
                    credentials=credentials, fact_id=fact_id
                )
                print(f"  Updated fact: {updated_fact}")
            else:
                print("✗ Failed to update fact")

            # Delete a different fact
            if len(fact_ids) > 1:
                delete_id = fact_ids[1]
                print(f"\n3. Deleting fact: {delete_id}")
                print("-" * 80)

                deleted = storage.delete_fact(
                    credentials=credentials, fact_id=delete_id
                )

                if deleted:
                    print("✓ Fact deleted successfully")

                    # Verify deletion
                    deleted_fact = storage.get_fact_by_id(
                        credentials=credentials, fact_id=delete_id
                    )
                    if deleted_fact is None:
                        print("✓ Fact no longer exists in database")
                else:
                    print("✗ Failed to delete fact")
        else:
            print("✗ Fact not found")

    print()

    # ==========================================================================
    # Example 5: Processed Messages Tracking (Batch Operations)
    # ==========================================================================
    print("=" * 80)
    print("Example 5: Processed Messages Tracking (Batch)")
    print("=" * 80)

    test_message_ids = [
        "msg_001_user_question",
        "msg_002_ai_response",
        "msg_003_user_followup",
        "msg_004_ai_response",
    ]

    # Check which messages have been processed (batch)
    print("\n1. Checking processed status for multiple messages...")
    print("-" * 80)
    processed = storage.check_processed_messages_batch(
        credentials=credentials, message_ids=test_message_ids
    )
    print(f"Already processed: {processed} ({len(processed)}/{len(test_message_ids)})")

    # Mark messages as processed (batch)
    print("\n2. Marking messages as processed (batch)...")
    print("-" * 80)
    message_data = [
        {"message_id": "msg_001_user_question", "thread_id": "thread_demo_123"},
        {"message_id": "msg_002_ai_response", "thread_id": "thread_demo_123"},
        {"message_id": "msg_003_user_followup", "thread_id": "thread_demo_123"},
    ]

    success = storage.mark_processed_messages_batch(
        credentials=credentials, message_data=message_data
    )

    if success:
        print(f"✓ Marked {len(message_data)} messages as processed")

        # Check again to verify
        processed_after = storage.check_processed_messages_batch(
            credentials=credentials, message_ids=test_message_ids
        )
        print(
            f"Now processed: {processed_after} ({len(processed_after)}/{len(test_message_ids)})"
        )
    else:
        print("✗ Failed to mark messages as processed")

    # Single message operations
    print("\n3. Single message operations...")
    print("-" * 80)

    single_msg_id = "msg_004_ai_response"
    is_processed = storage.check_processed_message(
        credentials=credentials, message_id=single_msg_id
    )
    print(f"Message {single_msg_id} processed: {is_processed}")

    if not is_processed:
        marked = storage.mark_processed_message(
            credentials=credentials,
            message_id=single_msg_id,
            thread_id="thread_demo_123",
        )
        if marked:
            print(f"✓ Marked {single_msg_id} as processed")

    print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
This example demonstrated:
✓ Inserting facts with embeddings
✓ Querying facts by vector similarity
✓ Filtering facts by namespace
✓ Getting, updating, and deleting facts
✓ Batch processing message tracking
✓ Single message processing operations

Next steps:
1. Replace mock embeddings with real embedding model
2. Integrate with your LangGraph agent
3. Use namespace hierarchies for better organization
4. Implement automatic fact extraction from conversations
5. Use processed message tracking to avoid duplicate work
    """)
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
