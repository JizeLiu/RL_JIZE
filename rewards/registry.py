"""Reward registry. A reward is a plain function registered by name.

Signature (sync or async)::

    def my_reward(solution_str: str, ground_truth, extra_info: dict, **args) -> float

Return a float in whatever range you like; the router combines rewards with per-route weights.
Add a new reward by creating ``rewards/<name>.py``, decorating the function with ``@register_reward("name")``
and listing the module in ``rewards/__init__.py``'s ``BUILTIN_MODULES`` (or ``RLX_REWARD_MODULES`` for
out-of-tree modules).
"""

from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}


def register_reward(name: str) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"reward {name!r} already registered to {_REGISTRY[name]!r}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_reward(name: str) -> Callable:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown reward {name!r}; registered: {sorted(_REGISTRY)}") from None


def registered_rewards() -> list[str]:
    return sorted(_REGISTRY)
