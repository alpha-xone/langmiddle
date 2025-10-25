"""Example demonstrating the is_tool_message utility function.

This example shows how the shared utility function can identify tool messages
across different middleware components.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langmiddle.utils import filter_tool_messages, is_tool_message


def demo_tool_message_detection():
    """Demonstrate tool message detection."""
    print("=" * 60)
    print("Tool Message Detection Examples")
    print("=" * 60)

    # Example 1: Regular tool message
    tool_msg = ToolMessage(content="Search result: Python is great", tool_call_id="call_123")
    print(f"\n1. ToolMessage: {is_tool_message(tool_msg)}")
    print(f"   Type: {tool_msg.type}")
    print(f"   Content: {tool_msg.content[:50]}...")

    # Example 2: AI message with tool calls
    ai_tool_msg = AIMessage(
        content="", response_metadata={"finish_reason": "tool_calls"}, tool_calls=[{"id": "call_123"}]
    )
    print(f"\n2. AI Message with tool_calls: {is_tool_message(ai_tool_msg)}")
    print(f"   Type: {ai_tool_msg.type}")
    print(f"   Finish reason: {ai_tool_msg.response_metadata.get('finish_reason')}")

    # Example 3: Regular AI message
    ai_msg = AIMessage(content="Here's the answer", response_metadata={"finish_reason": "stop"})
    print(f"\n3. Regular AI Message: {is_tool_message(ai_msg)}")
    print(f"   Type: {ai_msg.type}")
    print(f"   Finish reason: {ai_msg.response_metadata.get('finish_reason')}")

    # Example 4: Human message
    human_msg = HumanMessage(content="What is Python?")
    print(f"\n4. Human Message: {is_tool_message(human_msg)}")
    print(f"   Type: {human_msg.type}")
    print(f"   Content: {human_msg.content}")


def demo_message_filtering():
    """Demonstrate filtering tool messages from a conversation."""
    print("\n" + "=" * 60)
    print("Message Filtering Example")
    print("=" * 60)

    # Create a conversation with mixed message types
    messages = [
        HumanMessage(content="Search for Python tutorials"),
        AIMessage(content="", response_metadata={"finish_reason": "tool_calls"}),
        ToolMessage(content="Found 10 tutorials", tool_call_id="call_123"),
        AIMessage(content="I found 10 Python tutorials for you."),
        HumanMessage(content="Thanks! Show me the first one."),
        AIMessage(content="", response_metadata={"finish_reason": "tool_calls"}),
        ToolMessage(content="Tutorial 1: Introduction to Python", tool_call_id="call_456"),
        AIMessage(content="Here's the first tutorial: Introduction to Python"),
    ]

    print(f"\nOriginal conversation: {len(messages)} messages")
    for i, msg in enumerate(messages, 1):
        is_tool = "🔧 TOOL" if is_tool_message(msg) else "💬 USER"
        print(f"  {i}. {is_tool} - {msg.type}: {str(msg.content)[:40]}...")

    # Filter out tool messages
    filtered = filter_tool_messages(messages)

    print(f"\nFiltered conversation: {len(filtered)} messages (removed {len(messages) - len(filtered)})")
    for i, msg in enumerate(filtered, 1):
        print(f"  {i}. {msg.type}: {str(msg.content)[:50]}...")


def demo_shared_usage():
    """Demonstrate how both middleware use the same utility."""
    print("\n" + "=" * 60)
    print("Shared Usage Across Middleware")
    print("=" * 60)

    print("\n✅ Both ChatSaver (history.py) and MemoryExtractor (memory.py)")
    print("   now use the same is_tool_message() utility function!")

    print("\n📦 Available from langmiddle:")
    print("   from langmiddle.utils import is_tool_message, filter_tool_messages")

    print("\n📦 Also available from utils:")
    print("   from langmiddle.utils import is_tool_message, filter_tool_messages")

    print("\n🎯 Benefits:")
    print("   • Consistent tool message detection across all middleware")
    print("   • Single source of truth for tool message logic")
    print("   • Easier to maintain and update")
    print("   • Reusable in custom middleware")


if __name__ == "__main__":
    demo_tool_message_detection()
    demo_message_filtering()
    demo_shared_usage()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
