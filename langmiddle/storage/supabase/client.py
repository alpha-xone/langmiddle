import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

__all__ = ["supa_client"]

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
# connection_string = os.getenv("SUPABASE_CONNECTION_STRING")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL or ANON KEY not set in environment variables")

supa_client = create_client(supabase_url, supabase_key)
