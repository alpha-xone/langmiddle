"""
Example usage of ContextEngineer middleware for extracting and managing facts.

This example demonstrates how to:
1. Initialize ContextEngineer with a model and embedder
2. Use it as middleware in a LangGraph agent
3. Automatically extract facts from conversations
4. Query and update facts in the background
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from langmiddle.context import ContextEngineer
from langmiddle.history import ChatSaver, StorageContext

# Load environment variables
load_dotenv()


def main():
    """Demonstrate ContextEngineer middleware usage."""

    # =========================================================================
    # Example 1: Basic Setup with ContextEngineer
    # =========================================================================
    print("\n=== Example 1: Basic ContextEngineer Setup ===")

    # Initialize the context engineer
    context_engineer = ContextEngineer(
        model="openai:gpt-4o-mini",  # Model for fact extraction
        embedder="openai:text-embedding-3-small",  # Embedder for vector search
        backend="supabase",
        max_tokens_before_extraction=500,  # Extract after 500 tokens
        model_kwargs={"temperature": 0.1},
    )

    print("Context engineer initialized")
    print(f"  Model: {context_engineer.model.__class__.__name__}")
    print(f"  Embedder: {context_engineer.embedder.__class__.__name__}")
    print(f"  Backend: {context_engineer.backend}")

    # =========================================================================
    # Example 2: Using with LangGraph Agent
    # =========================================================================
    print("\n=== Example 2: Using ContextEngineer in Agent ===")

    # Create agent with multiple middleware
    from langchain.agents import create_agent

    agent = create_agent(
        model="openai:gpt-4o",
        tools=[],  # Add your tools here
        context_schema=StorageContext,
        middleware=[
            context_engineer,  # Extract facts
            ChatSaver(backend="supabase"),  # Save chat history
        ],
    )

    # Example conversation context
    context = StorageContext(
        user_id="user-123",
        thread_id="thread-456",
        auth_token="your-jwt-token",  # For Supabase RLS
    )

    # Example 1: User shares preferences
    print("\nConversation 1: User shares preferences")
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hi! I'm a Python developer and I prefer using "
                               "functional programming patterns. I really enjoy "
                               "working with asyncio and type hints."
                }
            ]
        },
        context=context,
    )
    print(f"Agent: {response['messages'][-1].content[:100]}...")

    # Example 2: User shares more information
    print("\nConversation 2: User shares project details")
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm currently building a RAG system using LangChain "
                               "and Supabase for vector storage. I prefer OpenAI's "
                               "GPT-4o for the main model."
                }
            ]
        },
        context=context,
    )
    print(f"Agent: {response['messages'][-1].content[:100]}...")

    # Example 3: Facts should now be available for future conversations
    print("\nConversation 3: Agent should remember preferences")
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What do you know about my preferences and projects?"
                }
            ]
        },
        context=context,
    )
    print(f"Agent: {response['messages'][-1].content[:200]}...")

    # =========================================================================
    # Example 3: Custom Extraction Prompts
    # =========================================================================
    print("\n=== Example 3: Custom Extraction Prompts ===")

    custom_extraction_prompt = """
    You are analyzing a conversation to extract key facts about the user.
    Focus specifically on:
    - Professional information (job, skills, experience)
    - Technical preferences (languages, frameworks, tools)
    - Current projects and goals

    Extract facts from the following messages:
    {messages}

    Return structured facts with high confidence levels.
    """

    custom_context_engineer = ContextEngineer(
        model="openai:gpt-4o-mini",
        embedder="openai:text-embedding-3-small",
        backend="supabase",
        extraction_prompt=custom_extraction_prompt,
        max_tokens_before_extraction=None,  # Extract on every completion
    )

    print("Custom context engineer initialized with specialized extraction")
    print(f"  Extraction count: {custom_context_engineer._extraction_count}")

    # =========================================================================
    # Example 4: Token Threshold Configuration
    # =========================================================================
    print("\n=== Example 4: Token Threshold Configuration ===")

    # Extract only when conversation exceeds 1000 tokens
    large_threshold_engineer = ContextEngineer(
        model="openai:gpt-4o-mini",
        embedder="openai:text-embedding-3-small",
        backend="supabase",
        max_tokens_before_extraction=1000,
    )

    print("Configured to extract only after 1000 tokens")
    print(f"  Threshold: {large_threshold_engineer.max_tokens_before_extraction}")

    # Extract on every completion (None threshold)
    always_extract_engineer = ContextEngineer(
        model="openai:gpt-4o-mini",
        embedder="openai:text-embedding-3-small",
        backend="supabase",
        max_tokens_before_extraction=None,
    )

    print("Configured to extract on every agent completion")
    print(f"  Threshold: {always_extract_engineer.max_tokens_before_extraction}")

    # =========================================================================
    # Example 5: Direct Storage Backend Usage (Advanced)
    # =========================================================================
    print("\n=== Example 5: Direct Backend Access ===")

    if context_engineer.storage:
        # You can also directly query facts from the backend
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        query_text = "What programming languages does the user prefer?"
        query_vec = embedder.embed_query(query_text)

        results = context_engineer.storage.backend.query_facts(
            query_embedding=query_vec,
            user_id="user-123",
            model_dimension=len(query_vec),
            match_threshold=0.7,
            match_count=5,
        )

        print(f"Found {len(results)} relevant facts:")
        for fact in results:
            print(f"  - {fact['content']} (similarity: {fact.get('similarity', 0):.2f})")

    # =========================================================================
    # Example 6: Monitoring Extraction Statistics
    # =========================================================================
    print("\n=== Example 6: Extraction Statistics ===")

    print(f"Total extractions performed: {context_engineer._extraction_count}")
    print(f"Logged messages count: {len(context_engineer._logged_messages)}")

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    # Note: Make sure you have set up your environment variables:
    # - OPENAI_API_KEY: Your OpenAI API key
    # - SUPABASE_URL: Your Supabase project URL
    # - SUPABASE_ANON_KEY: Your Supabase anonymous key
    # - SUPABASE_CONNECTION_STRING: For initial table creation (if needed)

    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up your .env file with required credentials")
        print("2. Run the vector.sql schema on your Supabase database")
        print("3. Installed required packages: pip install langmiddle[supabase]")
