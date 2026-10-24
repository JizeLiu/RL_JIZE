"""Route a sample to one or more registered rewards and combine them.

Routing table (``rewards/routes.yaml``, override with ``RLX_REWARD_ROUTES=/path.yaml``)::

    routes:
      - match: "math/*"                       # fnmatch against data_source; first match wins
        rewards:
          - {name: math_verify, weight: 1.0}
    fallback: verl                            # unmatched data_source -> verl's built-in scorers

The returned dict always has the same keys for every sample (VERL derives the logged extra-info keys
from the first sample): ``score`` (weighted sum), ``acc`` (1.0 if score > half of the route's total
weight) and ``reward/<name>`` for every reward referenced anywhere in the table (0.0 if unused).
"""

from __future__ import annotations

import asyncio
import fnmatch
import importlib
import inspect
import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml

from rewards.registry import get_reward, registered_rewards

logger = logging.getLogger(__name__)

BUILTIN_MODULES = ["rewards.math_reward", "rewards.instruction_following", "rewards.llm_judge"]
DEFAULT_ROUTES = Path(__file__).with_name("routes.yaml")


@lru_cache(maxsize=1)
def _load():
    for mod in BUILTIN_MODULES + [m for m in os.environ.get("RLX_REWARD_MODULES", "").split(",") if m]:
        importlib.import_module(mod)
    path = Path(os.environ.get("RLX_REWARD_ROUTES", DEFAULT_ROUTES))
    cfg = yaml.safe_load(path.read_text()) or {}
    routes = cfg.get("routes", [])
    names = set()
    for route in routes:
        for r in route["rewards"]:
            get_reward(r["name"])  # fail fast on typos
            names.add(r["name"])
    return routes, cfg.get("fallback", "verl"), sorted(names)


def reset_cache() -> None:
    _load.cache_clear()


def _find_route(data_source: str, routes: list[dict]):
    for route in routes:
        if fnmatch.fnmatch(data_source, route["match"]):
            return route
    return None


async def _call(fn, solution_str, ground_truth, extra_info, args):
    if inspect.iscoroutinefunction(fn):
        return await fn(solution_str, ground_truth, extra_info, **args)
    return await asyncio.to_thread(fn, solution_str, ground_truth, extra_info, **args)


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Entry point for ``reward.custom_reward_function`` (async is supported by VERL's reward loop)."""
    routes, fallback, names = _load()
    route = _find_route(str(data_source), routes)
    keys = {f"reward/{n}": 0.0 for n in names}

    if route is None:
        if fallback != "verl":
            raise KeyError(f"no reward route for data_source={data_source!r} and fallback={fallback!r}")
        from verl.utils.reward_score import default_compute_score

        res = await asyncio.to_thread(
            default_compute_score, data_source, solution_str, ground_truth, extra_info
        )
        score = float(res["score"] if isinstance(res, dict) else res)
        return {"score": score, "acc": float(score > 0.5), **keys}

    specs = route["rewards"]
    results = await asyncio.gather(
        *[_call(get_reward(s["name"]), solution_str, ground_truth, extra_info or {}, s.get("args", {})) for s in specs]
    )
    total_weight = sum(s.get("weight", 1.0) for s in specs)
    score = 0.0
    for spec, value in zip(specs, results, strict=True):
        keys[f"reward/{spec['name']}"] = float(value)
        score += spec.get("weight", 1.0) * float(value)
    return {"score": score, "acc": float(score > 0.5 * total_weight), **keys}


__all__ = ["compute_score", "reset_cache", "registered_rewards"]
