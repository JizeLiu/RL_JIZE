"""Correctness check used by the data tools. Defaults to the same router used in RL training
(``rewards.dispatch``) so "correct" means exactly what the trainer will reward."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

# (record, response) -> bool
VerifyFn = Callable[[dict, str], Awaitable[bool]]


async def reward_router_verify(record: dict, response: str) -> bool:
    from rewards.dispatch import compute_score

    extra = dict(record.get("extra_info") or {})
    extra.setdefault("question", record.get("question", ""))
    if "constraints" in record:
        extra.setdefault("constraints", record["constraints"])
    out = await compute_score(
        data_source=record.get("data_source", "unknown"),
        solution_str=response,
        ground_truth=record.get("ground_truth"),
        extra_info=extra,
    )
    return bool(out["acc"] > 0.5)
