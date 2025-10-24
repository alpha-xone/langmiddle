# Supabase Setup Guide

This guide explains how to set up Supabase storage backend for langmiddle.

## Quick Start (Recommended)

For the easiest setup, langmiddle can automatically create the required tables on first use.

### Step 1: Get Your Supabase Credentials

1. Go to your [Supabase project dashboard](https://app.supabase.com)
2. Navigate to **Settings** > **API**
3. Copy your:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **anon/public key**

### Step 2: Get the Connection String (One-time Setup Only)

For automatic table creation, you need the direct database connection string:

1. Go to **Settings** > **Database**
2. Scroll to **Connection string**
3. Select **Direct connection** (not Transaction or Session)
4. Copy the connection string (format: `postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres`)

### Step 3: Set Up Environment Variables (Optional)

Create a `.env` file in your project root:

```env
# Required for day-to-day usage
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here

# Only needed for first-time table creation
SUPABASE_CONNECTION_STRING=postgresql://postgres:password@db.xxxxx.supabase.co:5432/postgres
```

### Step 4: Initialize with Auto-Create Tables

```python
from langmiddle.storage import ChatStorage

# First-time setup: automatically create tables
storage = ChatStorage.create(
    "supabase",
    auto_create_tables=True,
    # Credentials can be loaded from .env or passed explicitly:
    supabase_url="https://xxxxx.supabase.co",
    supabase_key="your_anon_key",
    connection_string="postgresql://postgres:password@db.xxxxx.supabase.co:5432/postgres"
)

print("✓ Tables created successfully!")
```

### Step 5: Normal Day-to-Day Usage

After tables are created, you don't need the connection string anymore:

```python
from langmiddle import ChatSaver

# Simple - loads from .env
middleware = ChatSaver(backend="supabase")

# Or with explicit credentials
middleware = ChatSaver(
    backend="supabase",
    supabase_url="https://xxxxx.supabase.co",
    supabase_key="your_anon_key"
)
```

## Manual Setup (Alternative)

If you prefer to create tables manually or the auto-create fails:

### Option 1: Using Supabase SQL Editor

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Create a new query
4. Copy the contents from these files in order:
   - First: `langmiddle/storage/supabase/chat_threads.sql`
   - Then: `langmiddle/storage/supabase/chat_messages.sql`
5. Run each query

### Option 2: Using the Schema Files Directly

The SQL schema files are included in the package at:
```
langmiddle/storage/supabase/
  ├── chat_threads.sql
  └── chat_messages.sql
```

You can execute these using any PostgreSQL client or the Supabase CLI.

## Balancing Setup and Day-to-Day Use

The design philosophy:

1. **First-time setup**: Use `auto_create_tables=True` with `connection_string`
   - Tables are created automatically
   - Connection string is only needed once
   - Safe to run multiple times (idempotent)

2. **Normal usage**: Just use `supabase_url` and `supabase_key`
   - No connection string needed
   - Works with RLS (Row Level Security)
   - Can be loaded from environment variables

3. **In `.env` file**:
   ```env
   # Always needed
   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_ANON_KEY=your_anon_key

   # Only for initial setup (can remove after tables are created)
   SUPABASE_CONNECTION_STRING=postgresql://...
   ```

## Troubleshooting

### "psycopg2 is required for automatic table creation"

Install the Supabase extras which includes psycopg2:

```bash
pip install langmiddle[supabase]
```

### "Supabase tables already exist, skipping creation"

This is normal! The tables have already been created. You can remove `auto_create_tables=True` from your code.

### Connection String Not Working

Make sure you're using the **Direct connection** string (port 5432), not the **Transaction** or **Session** pooler strings.

### Manual Table Creation

If auto-creation fails, you can always create tables manually using the SQL Editor in Supabase dashboard. The SQL files are in the package under `storage/supabase/`.

## Security Best Practices

1. **Never commit** your connection string to version control
2. **Use environment variables** for all credentials
3. **Remove connection string** from `.env` after initial setup
4. **Enable RLS** (Row Level Security) on your tables in production
5. **Use the anon key** for client-side access (not the service role key)

## Example Project Structure

```
my-project/
├── .env                    # Credentials (gitignored)
├── .gitignore             # Include .env
├── setup_supabase.py      # One-time setup script
└── main.py                # Your application
```

**setup_supabase.py** (run once):
```python
from langmiddle.storage import ChatStorage

# Create tables
storage = ChatStorage.create("supabase", auto_create_tables=True)
print("✓ Supabase setup complete!")
```

**main.py** (your app):
```python
from langmiddle import ChatSaver

# Normal usage - no setup code needed
middleware = ChatSaver(backend="supabase")
```
