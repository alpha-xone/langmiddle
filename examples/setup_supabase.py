"""
One-time Supabase setup script.

This script creates the necessary database tables in your Supabase project.
You only need to run this once when setting up a new Supabase project.

Usage:
    python setup_supabase.py

Requirements:
    1. pip install langmiddle[supabase]
    2. Create a .env file with your Supabase credentials
"""

import os
import sys

try:
    from dotenv import load_dotenv
    from langmiddle.storage import ChatStorage
except ImportError:
    print("❌ Error: Required packages not installed")
    print("   Run: pip install langmiddle[supabase]")
    sys.exit(1)

# Load environment variables
load_dotenv()

print("=" * 60)
print("Supabase Setup for langmiddle")
print("=" * 60)

# Check for required environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
connection_string = os.getenv("SUPABASE_CONNECTION_STRING")

print("\n📋 Checking credentials...")

if not supabase_url:
    print("❌ SUPABASE_URL not found in environment")
    print("   Add it to your .env file")
    sys.exit(1)
print(f"✓ SUPABASE_URL: {supabase_url[:30]}...")

if not supabase_key:
    print("❌ SUPABASE_ANON_KEY not found in environment")
    print("   Add it to your .env file")
    sys.exit(1)
print(f"✓ SUPABASE_ANON_KEY: {supabase_key[:20]}...")

if not connection_string:
    print("\n⚠️  SUPABASE_CONNECTION_STRING not found")
    print("   This is required for automatic table creation.")
    print("\n   To get your connection string:")
    print("   1. Go to your Supabase project dashboard")
    print("   2. Settings > Database")
    print("   3. Copy the 'Direct connection' string")
    print("   4. Add it to your .env file as SUPABASE_CONNECTION_STRING")
    print("\n   Or create tables manually using the SQL files in:")
    print("   langmiddle/storage/supabase/*.sql")
    sys.exit(1)
print(f"✓ SUPABASE_CONNECTION_STRING: {connection_string[:40]}...")

# Create tables
print("\n🔨 Creating database tables...")
print("   This is safe to run multiple times (idempotent)")

try:
    storage = ChatStorage.create(
        "supabase",
        auto_create_tables=True
    )

    print("\n✅ Setup complete!")
    print("\n📝 Next steps:")
    print("   1. (Optional) Remove SUPABASE_CONNECTION_STRING from .env")
    print("   2. Use the middleware in your code:")
    print("\n   from langmiddle import ChatSaver")
    print("   middleware = ChatSaver(backend='supabase')")
    print("\n   That's it! The middleware will use SUPABASE_URL and SUPABASE_ANON_KEY")

except Exception as e:
    print(f"\n❌ Setup failed: {e}")
    print("\n💡 Alternative: Create tables manually")
    print("   1. Go to Supabase SQL Editor")
    print("   2. Run the SQL files from: langmiddle/storage/supabase/")
    print("      - First: chat_threads.sql")
    print("      - Then: chat_messages.sql")
    sys.exit(1)

print("\n" + "=" * 60)
