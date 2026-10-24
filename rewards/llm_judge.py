"""LLM-as-judge reward over an OpenAI-compatible chat endpoint (vLLM / SGLang / hosted API).

Configuration (environment, so the same code runs on every node)::

    RLX_JUDGE_BASE_URL     e.g. http://judge-node:8000/v1
    RLX_JUDGE_MODEL        model name served at that endpoint
    RLX_JUDGE_API_KEY      default "EMPTY"
    RLX_JUDGE_CONCURRENCY  max in-flight requests per reward worker (default 32)
    RLX_JUDGE_TIMEOUT      seconds per request (default 120)

The judge is asked to answer with ``Score: <number>`` on the last line; ``scale`` is the maximum
(10 -> divides by 10). Any failure (timeout, parse error) returns ``fail_score`` and is logged, so a
flaky judge degrades a batch instead of crashing the run.

Question text is read from ``extra_info["question"]`` and the optional reference from
``ground_truth`` (see data_process/SCHEMA.md).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable

from rewards.registry import register_reward
from rewards.text_utils import strip_thinking

logger = logging.getLogger(__name__)

RUBRICS = {
    "helpfulness": (
        "Rate how well the response answers the user's request: correctness, completeness, clarity and "
        "following the user's constraints."
    ),
    "reference": (
        "Rate whether the response is consistent with the reference answer. Full marks only if the final "
        "answer is equivalent to the reference."
    ),
}

PROMPT = """You are a strict grader.

[Question]
{question}

{reference_block}[Response]
{response}

[Rubric]
{rubric}

Give an integer score from 0 to {scale}. Briefly justify, then end with a final line of the form
"Score: <integer>"."""

ChatFn = Callable[[str], Awaitable[str]]
_chat_override: ChatFn | None = None
_semaphore: asyncio.Semaphore | None = None
_client = None


def set_chat_fn(fn: ChatFn | None) -> None:
    """Inject a custom async ``prompt -> text`` function (unit tests, non-OpenAI backends)."""
    global _chat_override
    _chat_override = fn


async def _chat(prompt: str) -> str:
    global _semaphore, _client
    if _chat_override is not None:
        return await _chat_override(prompt)
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(
            base_url=os.environ["RLX_JUDGE_BASE_URL"],
            api_key=os.environ.get("RLX_JUDGE_API_KEY", "EMPTY"),
            timeout=float(os.environ.get("RLX_JUDGE_TIMEOUT", "120")),
        )
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(int(os.environ.get("RLX_JUDGE_CONCURRENCY", "32")))
    async with _semaphore:
        resp = await _client.chat.completions.create(
            model=os.environ["RLX_JUDGE_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
    return resp.choices[0].message.content or ""


_SCORE_RE = re.compile(r"Score:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_score(text: str, scale: float) -> float | None:
    matches = _SCORE_RE.findall(text)
    if not matches:
        return None
    return min(max(float(matches[-1]) / scale, 0.0), 1.0)


@register_reward("llm_judge")
async def llm_judge_reward(
    solution_str: str,
    ground_truth=None,
    extra_info=None,
    *,
    rubric: str = "helpfulness",
    scale: float = 10.0,
    fail_score: float = 0.0,
    retries: int = 2,
    use_reference: bool = False,
) -> float:
    info = extra_info or {}
    question = info.get("question") or info.get("prompt") or ""
    reference_block = f"[Reference answer]\n{ground_truth}\n\n" if use_reference and ground_truth else ""
    prompt = PROMPT.format(
        question=question,
        reference_block=reference_block,
        response=strip_thinking(solution_str),
        rubric=RUBRICS.get(rubric, rubric),
        scale=int(scale),
    )
    for attempt in range(retries + 1):
        try:
            score = parse_score(await _chat(prompt), scale)
            if score is not None:
                return score
            logger.warning("judge reply had no parsable score (attempt %d)", attempt + 1)
        except Exception as exc:
            logger.warning("judge call failed (attempt %d): %s", attempt + 1, exc)
        await asyncio.sleep(min(2**attempt, 8))
    return fail_score
