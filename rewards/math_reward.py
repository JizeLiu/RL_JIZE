"""Verifiable math reward: compare the last ``\\boxed{}`` answer with the ground truth."""

from __future__ import annotations

import logging

from rewards.registry import register_reward
from rewards.text_utils import extract_boxed, normalize_math, strip_thinking

logger = logging.getLogger(__name__)

try:  # math-verify is optional; without it we fall back to normalised string equality
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify
except Exception:  # pragma: no cover - depends on environment
    _mv_parse = _mv_verify = None


def _symbolic_equal(pred: str, gold: str) -> bool:
    if _mv_parse is None:
        return False
    try:
        return bool(_mv_verify(_mv_parse(f"${gold}$"), _mv_parse(f"${pred}$")))
    except Exception as exc:  # parser / timeout errors must never crash a rollout batch
        logger.debug("math_verify failed: %s", exc)
        return False


@register_reward("math_verify")
def math_verify_reward(solution_str: str, ground_truth, extra_info=None, *, use_symbolic: bool = True) -> float:
    """1.0 if the boxed answer matches ``ground_truth``, else 0.0. No boxed answer -> 0.0."""
    pred = extract_boxed(strip_thinking(solution_str))
    if not pred:
        return 0.0
    gold = str(ground_truth)
    if normalize_math(pred) == normalize_math(gold):
        return 1.0
    return 1.0 if use_symbolic and _symbolic_equal(pred, gold) else 0.0
