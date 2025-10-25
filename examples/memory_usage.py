"""Example usage of MemoryExtractor middleware with LangGraph Store.

This example demonstrates how to use the MemoryExtractor middleware to
automatically extract and store memories from conversations.
"""

from langchain.agents.middleware import AgentState
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

from langmiddle.memory import MemoryExtractor


# Example 1: Basic usage with InMemoryStore
def basic_memory_extraction():
    """Basic example of memory extraction."""
    # Initialize the memory extractor - always extracts
    extractor = MemoryExtractor(
        model=ChatOpenAI(model="gpt-4"),
        namespace_prefix=["user", "123"],
    )

    # Create an in-memory store for demonstration
    store = InMemoryStore()

    # Simulate conversation state
    from langchain_core.messages import AIMessage, HumanMessage

    state = AgentState({
        "messages": [
            HumanMessage(content="Hi, my name is Alice and I love Python programming."),
            AIMessage(content="Nice to meet you Alice! Python is a great language."),
            HumanMessage(content="I'm working on a machine learning project."),
            AIMessage(content="That sounds interesting! What kind of ML project?"),
        ]
    })

    # Extract memories (in real usage, this is called automatically by middleware)
    from langgraph.runtime import Runtime

    runtime = Runtime()
    result = extractor.after_agent(state, runtime)

    if result and "memories" in result:
        # Store the extracted memories
        for memory in result["memories"]:
            store.put(
                namespace=memory["namespace"],
                key=memory["key"],
                value=memory["value"],
                ttl=memory.get("ttl"),
            )

        print(f"✅ Extracted and stored {len(result['memories'])} memories")

        # Retrieve and display stored memories
        print("\n📋 Stored Memories:")
        for memory in result["memories"]:
            item = store.get(memory["namespace"], memory["key"])
            if item:
                print(f"  • {memory['key']}: {item.value}")


# Example 2: Token-based extraction triggering
def token_based_extraction():
    """Example with token-based extraction control."""
    # Only extract when conversation reaches 4000 tokens
    extractor = MemoryExtractor(
        model=ChatOpenAI(model="gpt-4"),
        namespace_prefix=["user", "456"],
        max_tokens_before_extraction=4000,
        messages_to_extract_from=15,
    )

    print(f"✅ Token-based extractor configured: {extractor}")
    print("   Will extract when messages exceed 4000 tokens")
    print("   Will analyze last 15 messages")


# Example 3: Custom namespace and prompt
def custom_memory_extraction():
    """Example with custom configuration."""
    from langchain_core.prompts import ChatPromptTemplate

    # Custom prompt focusing on user preferences
    custom_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Extract user preferences and important facts from the conversation.
Focus specifically on:
- Personal information (name, location, etc.)
- Preferences (likes, dislikes, favorites)
- Professional information (job, skills, interests)

Format each memory with a clear key and structured value.""",
            ),
            ("user", "Conversation:\n{messages}"),
        ]
    )

    extractor = MemoryExtractor(
        model=ChatOpenAI(model="gpt-4"),
        namespace_prefix=["user", "preferences"],
        prompt_template=custom_prompt,
    )

    print(f"✅ Custom memory extractor configured: {extractor}")


# Example 4: Using with PostgresStore
def postgres_memory_extraction():
    """Example with PostgresStore for persistent storage."""
    from langgraph.store.postgres import AsyncPostgresStore

    async def extract_and_store():
        # Initialize PostgresStore
        conn_string = "postgresql://user:pass@localhost:5432/memories"

        async with AsyncPostgresStore.from_conn_string(conn_string) as store:
            await store.setup()

            # Initialize extractor
            extractor = MemoryExtractor(
                model=ChatOpenAI(model="gpt-4"),
                namespace_prefix=["user", "456"],
            )

            # Simulate conversation
            from langchain_core.messages import AIMessage, HumanMessage

            state = AgentState({
                "messages": [
                    HumanMessage(content="I prefer dark mode and Python 3.11+"),
                    AIMessage(content="Got it! I'll remember your preferences."),
                ]
            })

            # Extract memories
            from langgraph.runtime import Runtime

            runtime = Runtime()
            result = extractor.after_agent(state, runtime)

            if result and "memories" in result:
                # Store in PostgreSQL
                for memory in result["memories"]:
                    await store.aput(
                        namespace=memory["namespace"],
                        key=memory["key"],
                        value=memory["value"],
                        ttl=memory.get("ttl"),
                    )

                print(f"✅ Stored {len(result['memories'])} memories in PostgreSQL")

    # Run the async function
    import asyncio

    asyncio.run(extract_and_store())


# Example 5: Integration with LangGraph workflow
def langgraph_workflow_integration():
    """Example of using MemoryExtractor in a LangGraph workflow."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    # Define state
    # Create graph
    workflow = StateGraph(AgentState)

    # Add node
    def chatbot(state):
        model = ChatOpenAI(model="gpt-4")
        response = model.invoke(state.messages)
        return {"messages": [response]}

    workflow.add_node("chatbot", chatbot)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)

    # Compile with memory extractor middleware
    extractor = MemoryExtractor(
        model=ChatOpenAI(model="gpt-4"),
        namespace_prefix=["conversation", "thread-1"],
    )

    app = workflow.compile(
        checkpointer=MemorySaver(),
        # Note: In actual usage, you would add middleware here
        # middlewares=[extractor]
    )

    print(f"✅ LangGraph workflow configured with extractor: {extractor}")
    print(f"   App: {app}")


if __name__ == "__main__":
    print("=" * 60)
    print("Memory Extraction Middleware Examples")
    print("=" * 60)

    print("\n1️⃣  Basic Memory Extraction")
    print("-" * 60)
    basic_memory_extraction()

    print("\n2️⃣  Token-Based Extraction")
    print("-" * 60)
    token_based_extraction()

    print("\n3️⃣  Custom Configuration")
    print("-" * 60)
    custom_memory_extraction()

    print("\n4️⃣  PostgreSQL Storage (uncomment to run)")
    print("-" * 60)
    # postgres_memory_extraction()  # Uncomment if you have PostgreSQL setup

    print("\n5️⃣  LangGraph Workflow Integration")
    print("-" * 60)
    langgraph_workflow_integration()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
