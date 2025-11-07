"""Example: Using Refactored ContextEngineer with Builder Pattern.

This example demonstrates the new modular architecture of ContextEngineer,
showcasing the builder pattern for clean configuration and the ability
to use individual components independently for testing and customization.
"""

from langchain_core.messages import HumanMessage

from langmiddle import (  # Builder pattern (recommended); Individual components (for advanced usage)
    ContextEngineerBuilder,
    ContextFormatter,
    FactDeduplicator,
    GlobalContextRetriever,
    RelevanceFilter,
    RelevantFactsRetriever,
)


def example_1_basic_builder():
    """Example 1: Basic usage with builder pattern.

    This is the recommended way to create ContextEngineer with
    the new architecture - clean, type-safe, and fluent.
    """
    print("\n" + "=" * 80)
    print("Example 1: Basic Builder Pattern")
    print("=" * 80)

    # Create ContextEngineer using builder
    context_engineer = (
        ContextEngineerBuilder()
        .with_model("openai:gpt-4")
        .with_embedder("openai:text-embedding-3-small")
        .with_storage(
            "supabase",
            supabase_url="https://your-project.supabase.co",
            supabase_key="your-anon-key",
        )
        .with_config(
            enable_context_injection=True,
            max_recent_messages=10,
            match_threshold=0.70,
            enable_relevance_filter=True,
        )
        .build()
    )

    print("✅ ContextEngineer created with builder pattern")
    print(f"   - Model: {context_engineer.model.__class__.__name__}")
    print(f"   - Embedder: {context_engineer.embedder.__class__.__name__}")
    print(f"   - Backend: {context_engineer.backend}")
    print(f"   - Max recent messages: {context_engineer.max_recent_messages}")


def example_2_custom_configuration():
    """Example 2: Custom configuration options.

    The builder pattern makes it easy to customize various
    aspects of context engineering.
    """
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)

    context_engineer = (
        ContextEngineerBuilder()
        .with_model("openai:gpt-4o-mini")  # Use faster model
        .with_embedder("openai:text-embedding-3-small")
        .with_storage("supabase", supabase_url="...", supabase_key="...")
        .with_config(
            # Context injection settings
            enable_context_injection=True,
            max_recent_messages=5,  # Keep fewer messages
            # Retrieval settings
            match_threshold=0.75,  # Higher threshold for precision
            match_count=10,  # Retrieve fewer facts
            context_window=3,  # Consider fewer recent messages
            # Processing settings
            enable_deduplication=True,
            enable_relevance_filter=True,
            relevance_filter_threshold=5,  # Filter if >5 facts
            # Summarization settings
            summary_threshold=15,  # Summarize if >15 messages
            min_messages_to_summarize=5,
            # Extraction settings
            max_tokens_before_extraction=3000,
            enable_memory_extraction=True,
        )
        .build()
    )

    print("✅ ContextEngineer created with custom configuration")
    print(f"   - Match threshold: {context_engineer.match_threshold}")
    print(f"   - Match count: {context_engineer.match_count}")
    print(f"   - Relevance filter threshold: {context_engineer.relevance_filter_threshold}")


def example_3_component_inspection():
    """Example 3: Inspecting modular components.

    The new architecture exposes individual components for
    testing, inspection, and customization.
    """
    print("\n" + "=" * 80)
    print("Example 3: Component Inspection")
    print("=" * 80)

    context_engineer = (
        ContextEngineerBuilder()
        .with_model("openai:gpt-4")
        .with_embedder("openai:text-embedding-3-small")
        .with_storage("supabase", supabase_url="...", supabase_key="...")
        .build()
    )

    # Access components (internal, but useful for debugging)
    if context_engineer._components:
        print("✅ Modular components available:")
        print(f"   - Global retriever: {type(context_engineer._components.global_retriever).__name__}")
        print(f"   - Relevant retriever: {type(context_engineer._components.relevant_retriever).__name__}")
        print(f"   - Summarizer: {type(context_engineer._components.summarizer).__name__}")
        print(f"   - Deduplicator: {type(context_engineer._components.deduplicator).__name__}")
        print(f"   - Relevance filter: {type(context_engineer._components.relevance_filter).__name__}")
        print(f"   - Formatter: {type(context_engineer._components.formatter).__name__}")
        print(f"   - Queue builder: {type(context_engineer._components.queue_builder).__name__}")
    else:
        print("⚠️  Using legacy implementation (no components)")


def example_4_individual_components():
    """Example 4: Using individual components independently.

    For advanced users who want to use components outside
    of ContextEngineer (e.g., for testing or custom pipelines).
    """
    print("\n" + "=" * 80)
    print("Example 4: Individual Components")
    print("=" * 80)

    from langmiddle.context.models import Fact

    # Create standalone deduplicator
    deduplicator = FactDeduplicator()

    # Sample facts
    primary_facts = [
        Fact(id="1", content="User prefers Python", namespace=["user", "preferences"],
             confidence=0.9, intensity=0.8),
        Fact(id="2", content="User works at Tech Corp", namespace=["user", "professional"],
             confidence=0.95, intensity=0.7),
    ]

    secondary_facts = [
        Fact(id="1", content="User prefers Python", namespace=["user", "preferences"],
             confidence=0.9, intensity=0.8),  # Duplicate by ID
        Fact(id="3", content="User works at Tech Corp", namespace=["user", "professional"],
             confidence=0.95, intensity=0.7),  # Duplicate by content
        Fact(id="4", content="User knows JavaScript", namespace=["user", "skills"],
             confidence=0.8, intensity=0.6),  # Unique
    ]

    # Deduplicate
    unique_facts = deduplicator.deduplicate(primary_facts, secondary_facts)

    print("✅ Deduplication demonstration:")
    print(f"   - Primary facts: {len(primary_facts)}")
    print(f"   - Secondary facts (with duplicates): {len(secondary_facts)}")
    print(f"   - Unique facts after deduplication: {len(unique_facts)}")
    print(f"   - Remaining fact: {unique_facts[0].content}")

    # Create standalone formatter
    formatter = ContextFormatter()

    # Format facts
    formatted = formatter.format(unique_facts, "relevant")

    print("\n✅ Formatting demonstration:")
    print("Formatted output:")
    print(formatted)


def example_5_custom_retrievers():
    """Example 5: Extending with custom retrievers.

    The modular architecture makes it easy to create custom
    retrievers that implement the same interface.
    """
    print("\n" + "=" * 80)
    print("Example 5: Custom Retrievers")
    print("=" * 80)

    from langmiddle.context.models import Fact, RetrievalContext

    # Custom retriever that always returns test data
    class TestDataRetriever:
        """Custom retriever for testing."""

        def retrieve(self, context: RetrievalContext) -> list[Fact]:
            """Return test facts."""
            return [
                Fact(
                    id="test-1",
                    content=f"Test fact for user {context.user_id}",
                    namespace=["test"],
                    confidence=1.0,
                    intensity=1.0,
                )
            ]

    # Create instance
    test_retriever = TestDataRetriever()

    # Create a mock context
    from langmiddle.context.models import RetrievalContext

    mock_context = RetrievalContext(
        user_id="test-user",
        credentials={"mock": True},
        messages=[HumanMessage(content="Hello")],
    )

    # Retrieve
    facts = test_retriever.retrieve(mock_context)

    print("✅ Custom retriever demonstration:")
    print(f"   - Retrieved {len(facts)} test facts")
    print(f"   - Fact content: {facts[0].content}")
    print("\n💡 This pattern allows you to:")
    print("   - Create mock retrievers for testing")
    print("   - Implement custom retrieval strategies")
    print("   - Swap implementations without changing code")


def example_6_component_testing():
    """Example 6: Testing individual components.

    The modular architecture makes unit testing much easier.
    """
    print("\n" + "=" * 80)
    print("Example 6: Component Testing")
    print("=" * 80)

    from langmiddle.context.models import Fact

    # Test deduplicator
    def test_deduplicator():
        """Test deduplication logic."""
        dedup = FactDeduplicator()

        # Test case: ID-based deduplication
        primary = [Fact(id="1", content="Test", namespace=[], confidence=0.5, intensity=0.5)]
        secondary = [
            Fact(id="1", content="Test", namespace=[], confidence=0.5, intensity=0.5),
            Fact(id="2", content="Other", namespace=[], confidence=0.5, intensity=0.5),
        ]

        result = dedup.deduplicate(primary, secondary)
        assert len(result) == 1
        assert result[0].id == "2"

        # Test case: Content-based deduplication
        primary = [Fact(id="1", content="Test Content", namespace=[], confidence=0.5, intensity=0.5)]
        secondary = [
            Fact(id="2", content="test content", namespace=[], confidence=0.5, intensity=0.5),  # Same content, different case
            Fact(id="3", content="Different", namespace=[], confidence=0.5, intensity=0.5),
        ]

        result = dedup.deduplicate(primary, secondary)
        assert len(result) == 1
        assert result[0].id == "3"

        return True

    # Test formatter
    def test_formatter():
        """Test formatting logic."""
        formatter = ContextFormatter()

        facts = [
            Fact(
                id="1",
                content="User prefers Python",
                namespace=["user", "preferences"],
                confidence=0.9,
                intensity=0.8,
            )
        ]

        # Test global context formatting
        global_text = formatter.format(facts, "global")
        assert "User Profile" in global_text
        assert "User prefers Python" in global_text

        # Test relevant context formatting
        relevant_text = formatter.format(facts, "relevant")
        assert "Relevant Context" in relevant_text
        assert "User prefers Python" in relevant_text

        return True

    # Run tests
    print("Running unit tests...")
    try:
        assert test_deduplicator()
        print("✅ Deduplicator tests passed")

        assert test_formatter()
        print("✅ Formatter tests passed")

        print("\n💡 Benefits of modular testing:")
        print("   - No mocking required for pure logic")
        print("   - Fast test execution")
        print("   - Easy to understand and maintain")
        print("   - Clear test isolation")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")


def example_7_migration_comparison():
    """Example 7: Comparing old vs new patterns.

    Shows the difference between the old constructor-based
    approach and the new builder pattern.
    """
    print("\n" + "=" * 80)
    print("Example 7: Old vs New Patterns")
    print("=" * 80)

    print("OLD PATTERN (still supported):")
    print("""
    from langmiddle import ContextEngineer

    context_engineer = ContextEngineer(
        model="openai:gpt-4",
        embedder="openai:text-embedding-3-small",
        backend="supabase",
        extraction_prompt=CUSTOM_PROMPT,
        update_prompt=CUSTOM_UPDATE_PROMPT,
        max_tokens_before_extraction=2000,
        max_recent_messages=10,
        enable_context_injection=True,
        token_counter=count_tokens_approximately,
        model_kwargs={"temperature": 0.1},
        embedder_kwargs={},
        backend_kwargs={
            "supabase_url": "...",
            "supabase_key": "...",
        },
    )
    """)

    print("\nNEW PATTERN (recommended):")
    print("""
    from langmiddle import ContextEngineerBuilder

    context_engineer = (
        ContextEngineerBuilder()
        .with_model("openai:gpt-4")
        .with_embedder("openai:text-embedding-3-small")
        .with_storage("supabase", supabase_url="...", supabase_key="...")
        .with_config(
            max_tokens_before_extraction=2000,
            max_recent_messages=10,
            enable_context_injection=True,
        )
        .build()
    )
    """)

    print("\n✅ Benefits of new pattern:")
    print("   - More readable and maintainable")
    print("   - Type-safe configuration")
    print("   - Clear separation of concerns")
    print("   - Easy to extend with new components")
    print("   - Better for testing and debugging")
    print("\n⚠️  Backward compatibility:")
    print("   - Old pattern still works unchanged")
    print("   - Gradual migration path available")
    print("   - No breaking changes")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("REFACTORED CONTEXT ENGINEER EXAMPLES")
    print("=" * 80)
    print("\nDemonstrating the new modular architecture with:")
    print("- Builder pattern for clean configuration")
    print("- Individual components for testing and customization")
    print("- Better separation of concerns")
    print("- Improved maintainability and extensibility")

    try:
        example_1_basic_builder()
        example_2_custom_configuration()
        example_3_component_inspection()
        example_4_individual_components()
        example_5_custom_retrievers()
        example_6_component_testing()
        example_7_migration_comparison()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)

        print("\n💡 Key Takeaways:")
        print("   1. Use ContextEngineerBuilder for new code (cleaner, more maintainable)")
        print("   2. Old ContextEngineer(...) pattern still works (backward compatible)")
        print("   3. Individual components can be used independently (better testing)")
        print("   4. Easy to extend with custom retrievers/processors (flexible)")
        print("   5. ~1000 lines → 7 focused modules (better organization)")

        print("\n📚 Next Steps:")
        print("   - Read docs/025 REFACTORING_PROPOSAL.md for architecture details")
        print("   - Migrate existing code gradually using builder pattern")
        print("   - Create custom components for domain-specific needs")
        print("   - Write unit tests for individual components")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
