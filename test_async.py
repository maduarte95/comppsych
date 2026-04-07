"""Quick smoke test: async OpenAI client against OpenRouter."""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


async def test():
    client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    resp = await client.chat.completions.create(
        model="qwen/qwen3.5-flash-02-23",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say hi"}],
    )
    print("Response:", resp.choices[0].message.content)
    print("Model:", resp.model)
    print("Success!")


asyncio.run(test())
