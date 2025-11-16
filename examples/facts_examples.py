"""
Example demonstrating atomic query breaking for better fact retrieval.

This example shows how complex user queries are automatically broken down
into simpler atomic queries for more accurate fact matching.
"""

import json
from random import choice

from aini import aini
from langchain.chat_models import init_chat_model

from langmiddle.memory import facts_manager

global model


def break_query_example():
    """Demonstrate atomic query breaking."""
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

        atomic_queries = facts_manager.break_query_into_atomic(model=model, user_query=query)

        for i, atomic_query in enumerate(atomic_queries, 1):
            print(f"  {i}. {atomic_query}")

        print("-" * 80)


def cues_producer_example():
    """Demonstrate assembling the Cues Producer prompt."""
    facts = [
        "User's favorite color is blue",
        "User prefers email communication in the mornings",
        "User's preferred editor is Neovim",
    ]

    print("Cues Producer Prompts")
    print("=" * 80)

    for fact in facts:
        print(f"Fact: {fact}\n")
        print("\nRelevant Cues:")

        cues = facts_manager.generate_fact_cues(model=model, fact_content=fact)
        for i, cue in enumerate(cues, 1):
            print(f"  {i}. {cue}")

        print("-" * 80)


def fact_extractor_example():
    """Example of fact extraction from a list of messages."""

    with open("examples/generated_msgs.json", "r") as f:
        all_messages = json.load(f)

    if len(all_messages) == 0:
        print("No generated messages found in 'generated_msgs.json'.")
        return

    messages = choice(all_messages)['turns']

    print("Fact Extraction Example")
    print("=" * 80)

    extracted_facts = facts_manager.extract_facts(model=model, messages=messages)
    if not extracted_facts:
        print("  No facts extracted.")
    else:
        for i, fact in enumerate(extracted_facts, 1):
            print(f"  {i}. {fact}")

    print("-" * 80)


if __name__ == "__main__":

    # Choose a model
    available_models = aini("examples/llm")
    print("Choose a model below:")
    for i, m in enumerate(available_models.keys(), 1):
        print(f"{i}. {m}")
    model_choice = int(input("Enter the number of the model to use: ").strip()) - 1
    model_name = list(available_models.keys())[model_choice]
    model = init_chat_model(**available_models[model_name])

    # Let user choose which example to run
    print("Choose an example to run:")
    print("1. Atomic Query Breaking")
    print("2. Cues Producer Prompt Assembly")
    print("3. Facts Extraction")
    example_choice = input("Enter 1, 2, or 3: ").strip()
    if example_choice == "1":
        break_query_example()
    elif example_choice == "2":
        cues_producer_example()
    elif example_choice == "3":
        fact_extractor_example()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
