"""
Example usage of facts management with Supabase backend.

This example demonstrates how to:
1. Insert facts with embeddings
2. Query facts using vector similarity search
3. Update and delete facts
4. Handle dynamic vector table creation
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from langmiddle.storage.supabase_backend import SupabaseStorageBackend

# Load environment variables
load_dotenv()


def main():
    # Initialize Supabase backend
    # Make sure SUPABASE_URL and SUPABASE_ANON_KEY are set in .env
    storage = SupabaseStorageBackend(
        load_from_env=True,
        auto_create_tables=False  # Tables should already exist
    )

    # Initialize embeddings model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Example user ID (in production, get from authentication)
    user_id = "550e8400-e29b-41d4-a716-446655440000"

    # =========================================================================
    # Example 1: Insert facts with embeddings
    # =========================================================================
    print("\n=== Example 1: Inserting Facts ===")

    facts = [
        {
            "content": "User prefers Python over JavaScript for backend development",
            "namespace": ["user", "preferences", "programming"],
            "language": "en",
            "intensity": 0.9,
            "confidence": 0.95,
        },
        {
            "content": "User is working on a machine learning project using PyTorch",
            "namespace": ["user", "projects", "current"],
            "language": "en",
            "intensity": 0.8,
            "confidence": 0.9,
        },
        {
            "content": "User's favorite programming paradigm is functional programming",
            "namespace": ["user", "preferences", "programming"],
            "language": "en",
            "intensity": 0.7,
            "confidence": 0.85,
        },
    ]

    # Generate embeddings for facts
    fact_contents = [fact["content"] for fact in facts]
    fact_embeddings = embeddings_model.embed_documents(fact_contents)

    model_dimension = len(fact_embeddings[0])
    print(f"Embedding dimension: {model_dimension}")

    # Insert facts
    result = storage.insert_facts(
        user_id=user_id,
        facts=facts,
        embeddings=fact_embeddings,
        model_dimension=model_dimension,
    )

    print(f"Inserted {result['inserted_count']} facts")
    if result['errors']:
        print(f"Errors: {result['errors']}")

    fact_ids = result['fact_ids']
    print(f"Fact IDs: {fact_ids}")

    # =========================================================================
    # Example 2: Query facts using similarity search
    # =========================================================================
    print("\n=== Example 2: Querying Facts ===")

    query_text = "What programming languages does the user prefer?"
    query_embedding = embeddings_model.embed_query(query_text)

    results = storage.query_facts(
        query_embedding=query_embedding,
        user_id=user_id,
        model_dimension=model_dimension,
        match_threshold=0.7,
        match_count=5,
    )

    print(f"Found {len(results)} matching facts:")
    for i, fact in enumerate(results, 1):
        print(f"\n{i}. Similarity: {fact.get('similarity', 0):.3f}")
        print(f"   Content: {fact['content']}")
        print(f"   Namespace: {fact['namespace']}")
        print(f"   Confidence: {fact.get('confidence', 'N/A')}")

    # =========================================================================
    # Example 3: Query with namespace filtering
    # =========================================================================
    print("\n=== Example 3: Querying with Namespace Filter ===")

    results_filtered = storage.query_facts(
        query_embedding=query_embedding,
        user_id=user_id,
        model_dimension=model_dimension,
        match_threshold=0.5,
        match_count=10,
        filter_namespaces=[["user", "preferences", "programming"]],
    )

    print(f"Found {len(results_filtered)} facts in 'user.preferences.programming' namespace")

    # =========================================================================
    # Example 4: Get a specific fact by ID
    # =========================================================================
    if fact_ids:
        print("\n=== Example 4: Getting Fact by ID ===")

        fact = storage.get_fact_by_id(
            fact_id=fact_ids[0],
            user_id=user_id,
        )

        if fact:
            print(f"Retrieved fact: {fact['content']}")
            print(f"Namespace: {fact['namespace']}")

    # =========================================================================
    # Example 5: Update a fact
    # =========================================================================
    if fact_ids:
        print("\n=== Example 5: Updating a Fact ===")

        updates = {
            "content": "User strongly prefers Python over JavaScript for all development",
            "intensity": 1.0,
            "confidence": 1.0,
        }

        # Generate new embedding for updated content
        new_embedding = embeddings_model.embed_query(updates["content"])

        success = storage.update_fact(
            fact_id=fact_ids[0],
            user_id=user_id,
            updates=updates,
            embedding=new_embedding,
        )

        if success:
            print("Fact updated successfully")

            # Retrieve updated fact
            updated_fact = storage.get_fact_by_id(fact_ids[0], user_id)
            if updated_fact:
                print(f"Updated content: {updated_fact['content']}")
                print(f"Updated intensity: {updated_fact['intensity']}")

    # =========================================================================
    # Example 6: Delete a fact
    # =========================================================================
    if len(fact_ids) > 1:
        print("\n=== Example 6: Deleting a Fact ===")

        fact_to_delete = fact_ids[1]
        success = storage.delete_fact(
            fact_id=fact_to_delete,
            user_id=user_id,
        )

        if success:
            print(f"Successfully deleted fact {fact_to_delete}")

            # Verify deletion
            deleted_fact = storage.get_fact_by_id(fact_to_delete, user_id)
            if deleted_fact is None:
                print("Fact confirmed deleted")

    # =========================================================================
    # Example 7: Dynamic vector table creation
    # =========================================================================
    print("\n=== Example 7: Dynamic Vector Table Creation ===")

    # Try a different embedding model with different dimensions
    new_dimension = 768  # Example: different model dimension

    table_created = storage.get_or_create_embedding_table(new_dimension)
    if table_created:
        print(f"Embedding table for dimension {new_dimension} is ready")
    else:
        print(f"Failed to create/verify embedding table for dimension {new_dimension}")


if __name__ == "__main__":
    main()
