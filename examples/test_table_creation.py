"""
Test script to verify detailed table creation logging.

Run this to see the new detailed logs for table creation.
"""
from dotenv import load_dotenv

from langmiddle.storage import ChatStorage

# Load environment variables
load_dotenv()

print("=" * 80)
print("Testing Supabase Table Creation with Detailed Logging")
print("=" * 80)
print()

# Create storage with auto_create_tables and enable_facts
storage = ChatStorage.create(
    'supabase',
    auto_create_tables=True,
    enable_facts=True
)

print()
print("=" * 80)
print("Table creation completed - check logs above for details")
print("=" * 80)
