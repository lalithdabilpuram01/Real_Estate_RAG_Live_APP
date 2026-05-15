# validation.py (async version)

from groq import AsyncGroq
import asyncio


async def validate_groq_api_key_async(api_key):
    """Async validation for better performance"""
    if not api_key:
        return False, "API key is empty"

    cleaned_key = api_key.strip().strip("'\"")

    if len(cleaned_key) < 20:
        return False, "API key is too short"

    try:
        client = AsyncGroq(api_key=cleaned_key)

        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="llama-3.3-70b-versatile",
            max_tokens=5
        )

        return True, "API key is valid"

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            return False, "Invalid API key"
        else:
            return False, f"API error: {error_msg}"


def validate_groq_api_key(api_key):
    """Sync wrapper for async validation"""
    return asyncio.run(validate_groq_api_key_async(api_key))