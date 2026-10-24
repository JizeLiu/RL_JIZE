"""Async sampling against OpenAI-compatible endpoints (vLLM / SGLang servers started by Slurm)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# (messages, n, temperature, max_tokens) -> list of n completions
SampleFn = Callable[[list[dict], int, float, int], Awaitable[list[str]]]


def parse_endpoints(items: list[str]) -> dict[str, tuple[str, str]]:
    """``name=http://host:8000/v1[@served-model-name]`` -> {name: (base_url, model)}."""
    out = {}
    for item in items:
        name, _, rest = item.partition("=")
        if not rest:
            raise ValueError(f"--endpoint entries look like name=url[@model], got {item!r}")
        url, _, model = rest.partition("@")
        out[name] = (url, model or name)
    return out


def make_openai_sampler(base_url: str, model: str, api_key: str = "EMPTY", concurrency: int = 64, retries: int = 3,
                        timeout: float = 600.0) -> SampleFn:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    sem = asyncio.Semaphore(concurrency)

    async def sample(messages: list[dict], n: int, temperature: float, max_tokens: int) -> list[str]:
        for attempt in range(retries + 1):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=model, messages=messages, n=n, temperature=temperature, max_tokens=max_tokens
                    )
                return [c.message.content or "" for c in resp.choices]
            except Exception as exc:
                logger.warning("sampling failed on %s (attempt %d): %s", base_url, attempt + 1, exc)
                if attempt == retries:
                    raise
                await asyncio.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    return sample
