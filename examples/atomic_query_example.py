"""
Example demonstrating atomic query breaking for better fact retrieval.

This example shows how complex user queries are automatically broken down
into simpler atomic queries for more accurate fact matching.
"""

from langchain.chat_models import init_chat_model

from langmiddle.memory.facts_manager import break_query_into_atomic


def main():
    """Demonstrate atomic query breaking."""
    # Initialize model
    model = init_chat_model("deepseek-chat")

    # Example complex queries
    complex_queries = [
        "What's the difference between LangGraph and LangChain, and how can I use either with Supabase memory?",
        "Tell me about my favorite foods and what cities I've lived in",
        "What are my communication preferences and do I have any pets?",
    ]

    print("Atomic Query Breaking Examples")
    print("=" * 80)

    for query in complex_queries:
        print(f"\nOriginal Query:\n  {query}")
        print("\nAtomic Queries:")

        atomic_queries = break_query_into_atomic(model=model, user_query=query)

        for i, atomic_query in enumerate(atomic_queries, 1):
            print(f"  {i}. {atomic_query}")

        print("-" * 80)


if __name__ == "__main__":
    main()
