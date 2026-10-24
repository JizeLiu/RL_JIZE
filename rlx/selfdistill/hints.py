"""Privileged-context construction for self-distillation (pure python, no VERL / torch).

Two things live here:

* ``plan_hints``: for every rollout decide *which* hints its teacher passes will see
  (dataset ground truth and/or sibling rollouts of the same prompt).
* ``render_hint_messages``: turn a chat prompt + a hint into the teacher's chat prompt.

The template is deliberately identical for positive and negative hints (RLCSD Sec. 3.3), so any
style shift the hint induces is shared between branches and cancels in the contrast.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

DEFAULT_TEMPLATE = (
    "{problem}\n\n"
    "{reference_block}"
    "Correct final answer: {answer}\n\n"
    "The reference above is only provided as private guidance. Now solve the problem step by step "
    "in your own words, and put your final answer within \\boxed{{}}."
)
_REFERENCE_BLOCK = "--- Reference Solution ---\n{solution}\n--- End of Reference Solution ---\n\n"


@dataclass(frozen=True)
class HintSource:
    """Where a hint comes from: dataset ground truth, or a sibling rollout (row index in the batch)."""

    kind: str  # "gt" | "rollout"
    row: int | None = None


@dataclass
class HintPlan:
    positive: HintSource | None = None
    negatives: list[HintSource] = field(default_factory=list)


def extract_boxed(text: str) -> str:
    """Return the content of the last ``\\boxed{...}`` in ``text`` (balanced braces), or ``""``."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return ""
    i = start + len("\\boxed{")
    depth, begin = 1, i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[begin : i - 1].strip() if depth == 0 else ""


def render_hint_messages(
    messages: Sequence[dict],
    *,
    solution: str,
    answer: str,
    template: str = DEFAULT_TEMPLATE,
) -> list[dict]:
    """Rewrite the last user turn of ``messages`` so it carries the privileged reference.

    ``solution`` may be empty (answer-only hint); the reference block is then omitted.
    """
    msgs = [dict(m) for m in messages]
    for idx in range(len(msgs) - 1, -1, -1):
        if msgs[idx].get("role") == "user":
            reference_block = _REFERENCE_BLOCK.format(solution=solution.strip()) if solution.strip() else ""
            msgs[idx]["content"] = template.format(
                problem=msgs[idx]["content"], reference_block=reference_block, answer=answer
            )
            return msgs
    raise ValueError("render_hint_messages: no user message found in prompt")


def plan_hints(
    uids: Sequence[str],
    rewards: Sequence[float],
    *,
    positive_source: str = "gt",
    n_neg: int = 0,
    correct_threshold: float = 0.5,
    seed: int = 0,
) -> list[HintPlan]:
    """Choose hint sources for each rollout.

    Args:
        uids: group id (prompt id) of every row.
        rewards: scalar outcome reward of every row.
        positive_source: ``"gt"`` (dataset reference) or ``"rollout"`` (a correct sibling).
        n_neg: number of negative hints per row (RLCSD uses K > 0; RLSD / OPSD use 0). Negatives
            are incorrect siblings and never include the row itself.
        correct_threshold: reward strictly above this counts as correct.
        seed: RNG seed; pass something step-dependent for fresh samples every step.

    A row gets ``positive=None`` when no positive hint is available (rollout mode without a correct
    sibling); callers should leave such rows' advantage untouched.
    """
    if positive_source not in {"gt", "rollout"}:
        raise ValueError(f"positive_source must be 'gt' or 'rollout', got {positive_source!r}")
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)

    plans = [HintPlan() for _ in uids]
    for rows in groups.values():
        correct = [i for i in rows if rewards[i] > correct_threshold]
        wrong = [i for i in rows if rewards[i] <= correct_threshold]
        for i in rows:
            plan = plans[i]
            if positive_source == "gt":
                plan.positive = HintSource("gt")
            else:
                pool = [j for j in correct if j != i]
                plan.positive = HintSource("rollout", rng.choice(pool)) if pool else None
            if n_neg > 0:
                pool = [j for j in wrong if j != i]
                picked = rng.sample(pool, min(n_neg, len(pool)))
                plan.negatives = [HintSource("rollout", j) for j in picked]
    return plans
