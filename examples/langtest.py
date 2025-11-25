"""
LangGraph SDK Debug Tool with LangMiddle Authentication

This debug tool uses LangMiddle CLI for authentication.

Setup:
1. Register or login using LangMiddle CLI:
   $ langmiddle auth register
   OR
   $ langmiddle auth login

2. Credentials are automatically saved to ~/.langmiddle/credentials.json

3. Run this script - authentication is handled automatically

Usage:
    from langtest import stream, search_assistants, get_thread

    # Check authentication
    check_authentication()

    # Use any function - they automatically use saved credentials
    await stream("Hello!", assistant_id="my-assistant")
"""

import json
import os
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import jwt as pyjwt
import requests
from dotenv import load_dotenv
from langgraph_sdk import get_client
from langgraph_sdk.schema import StreamMode

load_dotenv()
load_dotenv(".env.test")

LG_URL = os.getenv("LANGGRAPH_URL", "http://127.0.0.1:2024")

client = get_client(url=LG_URL, api_key=os.getenv("LANGSMITH_API_KEY", ""))


def load_langmiddle_credentials() -> Optional[dict]:
    """
    Load credentials from ~/.langmiddle/credentials.json

    Returns:
        Dictionary with access_token, refresh_token, user_id, email
        None if not found or invalid
    """
    credentials_file = Path.home() / ".langmiddle" / "credentials.json"

    if not credentials_file.exists():
        print("❌ No credentials found. Please run: langmiddle auth login")
        return None

    try:
        with open(credentials_file, "r") as f:
            credentials = json.load(f)

        required_fields = ["access_token", "user_id"]
        if not all(field in credentials for field in required_fields):
            print("❌ Invalid credentials file. Please run: langmiddle auth login")
            return None

        return credentials
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return None


def get_access_headers(**kwargs) -> Optional[dict]:
    """
    Get authorization headers using LangMiddle credentials

    Returns:
        Dictionary with Authorization header
        None if credentials not available
    """
    credentials = load_langmiddle_credentials()
    # If credentials file is missing or invalid, try a top-level refresh once
    if not credentials:
        refreshed = refresh_langmiddle_token()
        if refreshed:
            credentials = refreshed
        else:
            return None

    access_token = credentials.get("access_token")

    # Helper: check expiry and attempt refresh if expired
    def is_token_expired(token: str) -> bool:
        try:
            payload = pyjwt.decode(token, options={"verify_signature": False})
            exp = payload.get("exp")
            if not exp:
                return False
            # Consider token expired if within 10 seconds of exp
            return time.time() >= (exp - 10)
        except Exception:
            # If we can't decode the token, assume it's expired so we try to refresh
            return True

    def refresh_access_token(creds: dict) -> Optional[dict]:
        # Use stored project_url and refresh_token to request a new access token
        project_url = creds.get("project_url") or os.getenv("LANGMIDDLE_PROJECT_URL")
        refresh_token = creds.get("refresh_token")
        if not project_url or not refresh_token:
            return None

        token_url = project_url.rstrip("/") + "/auth/v1/token"
        try:
            resp = requests.post(
                token_url,
                data={"grant_type": "refresh_token", "refresh_token": refresh_token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
        except Exception as e:
            print(f"❌ Error refreshing token: {e}")
            return None

        if resp.status_code not in (200, 201):
            print(f"❌ Token refresh failed ({resp.status_code}): {resp.text}")
            return None

        data = resp.json()
        # Merge returned tokens (Supabase-like response: access_token, refresh_token)
        creds["access_token"] = data.get("access_token", creds.get("access_token"))
        creds["refresh_token"] = data.get("refresh_token", creds.get("refresh_token"))

        # Persist updated credentials back to the credentials file
        try:
            credentials_file = Path.home() / ".langmiddle" / "credentials.json"
            with open(credentials_file, "w") as f:
                json.dump(creds, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: could not persist refreshed token: {e}")

        return creds

    # If we don't have an access token but have a refresh token, try refresh
    if not access_token and credentials.get("refresh_token"):
        refreshed = refresh_access_token(credentials)
        if refreshed:
            access_token = refreshed.get("access_token")

    # If token appears expired, try to refresh using refresh_token
    if access_token and is_token_expired(access_token):
        refreshed = refresh_access_token(credentials)
        if refreshed is None:
            print("❌ Access token expired and refresh failed. Please run: langmiddle auth login")
            return None

        access_token = refreshed.get("access_token")

    return {"Authorization": f"Bearer {access_token}"}


def check_authentication():
    """
    Check if user is authenticated with LangMiddle

    Prints helpful message if not authenticated
    """
    credentials = load_langmiddle_credentials()
    if not credentials:
        print("\n" + "=" * 60)
        print("Authentication Required")
        print("=" * 60)
        print("\nTo use this debug tool, authenticate with LangMiddle CLI:")
        print("\n  Option 1 - Register (first time):")
        print("    langmiddle auth register")
        print("\n  Option 2 - Login (existing account):")
        print("    langmiddle auth login")
        print("\nCredentials will be saved to: ~/.langmiddle/credentials.json")
        print("=" * 60 + "\n")
        return False

    print(f"✅ Authenticated as: {credentials.get('email', 'N/A')}")
    print(f"   User ID: {credentials['user_id']}")
    return True


def refresh_langmiddle_token(creds: Optional[dict] = None) -> Optional[dict]:
    """
    Top-level helper to refresh stored LangMiddle access token using the refresh_token.

    Returns the updated credentials dict on success, or None on failure.
    """
    credentials = creds or load_langmiddle_credentials()
    if not credentials:
        return None

    project_url = credentials.get("project_url") or os.getenv("LANGMIDDLE_PROJECT_URL")
    refresh_token = credentials.get("refresh_token")
    if not project_url or not refresh_token:
        return None

    token_url = project_url.rstrip("/") + "/auth/v1/token"
    try:
        resp = requests.post(
            token_url,
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
    except Exception as e:
        print(f"❌ Error refreshing token: {e}")
        return None

    if resp.status_code not in (200, 201):
        print(f"❌ Token refresh failed ({resp.status_code}): {resp.text}")
        return None

    data = resp.json()
    credentials["access_token"] = data.get("access_token", credentials.get("access_token"))
    credentials["refresh_token"] = data.get("refresh_token", credentials.get("refresh_token"))

    try:
        credentials_file = Path.home() / ".langmiddle" / "credentials.json"
        with open(credentials_file, "w") as f:
            json.dump(credentials, f, indent=2)
    except Exception as e:
        print(f"⚠️ Warning: could not persist refreshed token: {e}")

    return credentials


async def search_assistants(**kwargs):
    """Get assistants"""
    headers = get_access_headers(**kwargs)
    if not headers:
        raise Exception("Failed to get valid access headers")
    return await client.assistants.search(headers=headers, **kwargs)


async def get_assistant(assistant_id, **kwargs):
    """Get assistant by ID"""
    headers = get_access_headers(**kwargs)
    if not headers:
        raise Exception("Failed to get valid access headers")
    return await client.assistants.get(assistant_id=assistant_id, headers=headers, **kwargs)


async def search_threads(**kwargs):
    """Get threads"""
    headers = get_access_headers(**kwargs)
    if not headers:
        raise Exception("Failed to get valid access headers")
    return await client.threads.search(headers=headers, **kwargs)


async def get_thread(thread_id, **kwargs):
    """Get thread by ID"""
    headers = get_access_headers(**kwargs)
    if not headers:
        raise Exception("Failed to get valid access headers")
    return await client.threads.get(thread_id=thread_id, headers=headers, **kwargs)


async def create_thread(**kwargs):
    """Create thread"""
    headers = get_access_headers(**kwargs)
    if not headers:
        # Try to refresh stored tokens and re-acquire headers once
        refreshed = refresh_langmiddle_token()
        if refreshed:
            headers = get_access_headers(**kwargs)

    if not headers:
        raise Exception("Failed to get valid access headers")

    try:
        return await client.threads.create(headers=headers, **kwargs)
    except Exception as e:
        # If the failure looks like an authorization error, try refreshing once and retry
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        msg = str(e)
        if status == 401 or "401" in msg or "unauthorized" in msg.lower() or "authorization" in msg.lower():
            refreshed = refresh_langmiddle_token()
            if refreshed:
                headers = get_access_headers(**kwargs)
                if headers:
                    return await client.threads.create(headers=headers, **kwargs)
        # re-raise original exception if retry not performed or still failing
        raise


async def delete_thread(thread_id, **kwargs):
    """Delete thread"""
    headers = get_access_headers(**kwargs)
    if not headers:
        raise Exception("Failed to get valid access headers")
    return await client.threads.delete(thread_id=thread_id, headers=headers, **kwargs)


async def store_put(namespace, key, value, **kwargs):
    """Store a memory item in the cross-thread memory store"""
    headers = get_access_headers(**kwargs)
    if not headers:
        raise Exception("Failed to get valid access headers")
    return await client.store.put_item(tuple(namespace), key, value, headers=headers, **kwargs)


async def store_get(namespace, key, **kwargs):
    """Get a memory item from the cross-thread memory store"""
    headers = get_access_headers(**kwargs)
    if not headers:
        raise Exception("Failed to get valid access headers")
    return await client.store.get_item(tuple(namespace), key, headers=headers, **kwargs)


async def store_search(namespace, **kwargs):
    """Search memory items using semantic search in the cross-thread memory store"""
    headers = get_access_headers(**kwargs)
    if not headers:
        raise Exception("Failed to get valid access headers")
    return await client.store.search_items(tuple(namespace), headers=headers, **kwargs)


async def store_delete(namespace, key, **kwargs):
    """Delete a memory item from the cross-thread memory store"""
    headers = get_access_headers(**kwargs)
    if not headers:
        raise Exception("Failed to get valid access headers")
    return await client.store.delete_item(tuple(namespace), key, headers=headers, **kwargs)


async def stream(
        msg: str,
        thread_id=None,
        assistant_id=None,
        stream_mode: StreamMode = "events",
        **kwargs,
):
    """Stream responses from a thread"""
    import jwt as ajwt

    headers = get_access_headers()
    if not headers:
        raise Exception("Failed to get valid access headers. Run: langmiddle auth login")
    headers["Accept"] = "application/json"
    headers["Content-Type"] = "application/json"

    chunks = []
    resolved_assistant_id = assistant_id if assistant_id is not None else os.getenv("ASSISTANT_ID", "")
    if not isinstance(resolved_assistant_id, str) or not resolved_assistant_id:
        raise ValueError("assistant_id must be a non-empty string")

    if thread_id is None:
        thread_id = str(uuid4())

    # Get credentials for context
    credentials = load_langmiddle_credentials()
    token = headers.get("Authorization", " ").split(" ")[1]
    context = {
        "thread_id": thread_id,
        "user_id": credentials["user_id"] if credentials else ajwt.decode(token, options={"verify_signature": False}).get("sub", " "),
        "auth_token": token,
    }

    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        input={"messages": [{"role": "user", "content": msg}]},
        headers=headers,
        if_not_exists="create",
        stream_mode=stream_mode,
        context=context if assistant_id != "play" else None,
        **kwargs,
    ):
        chunks.append(chunk)
        if stream_mode == "events":
            if chunk.data.get("event", "") == "on_chat_model_stream":
                print(chunk.data["data"]["chunk"]["content"], end="", flush=True)

    return chunks


if __name__ == "__main__":
    """
    Interactive test mode

    Run this script directly to test authentication and basic functionality
    """
    print("\n" + "=" * 60)
    print("LangGraph SDK Debug Tool - LangMiddle Authentication")
    print("=" * 60)

    # Check authentication
    if not check_authentication():
        exit(1)

    print("\nAuthentication successful! Ready to use debug functions.")
    print("\nAvailable functions:")
    print("  - search_assistants()")
    print("  - get_assistant(assistant_id)")
    print("  - search_threads()")
    print("  - get_thread(thread_id)")
    print("  - create_thread()")
    print("  - delete_thread(thread_id)")
    print("  - store_put(namespace, key, value)")
    print("  - store_get(namespace, key)")
    print("  - store_search(namespace)")
    print("  - store_delete(namespace, key)")
    print("  - stream(msg, thread_id, assistant_id)")
    print("\nImport this module to use these functions in your code.")
    print("=" * 60 + "\n")
