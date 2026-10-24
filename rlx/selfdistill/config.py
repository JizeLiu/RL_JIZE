"""Hydra-facing config for the self-distillation trainer.

All keys live under a top-level ``selfdistill`` node that VERL does not know about, so recipes pass
them with Hydra's append syntax, e.g. ``+selfdistill.method=rlsd``. Nothing in VERL's own config
tree is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

METHODS = ("none", "opsd", "rlsd", "rlcsd")


@dataclass
class SelfDistillConfig:
    method: str = "none"
    # Where the positive hint comes from: dataset reference ("gt") or a correct sibling rollout.
    positive_source: str = "gt"
    # extra_info[<key>] holds the reference reasoning trace; reward_model.ground_truth holds the answer.
    gt_solution_key: str = "solution"
    template_path: str | None = None
    correct_threshold: float = 0.5
    max_hint_tokens: int = 2048
    # "actor": current policy (online self-teacher). "ref": frozen initial policy (needs a ref model,
    # i.e. actor.use_kl_loss=true with kl_loss_coef=0). RLCSD defaults to "ref" as in the paper.
    teacher_mode: str = "actor"
    seed: int = 0

    # RLSD
    eps_w: float = 0.2
    lam_start: float = 0.5
    lam_end: float = 0.0
    lam_decay_steps: int = 50

    # OPSD
    opsd_clip: float = 5.0

    # RLCSD
    n_neg: int = 4
    rlcsd_lam: float = 0.5
    rlcsd_tau: float = 1.3
    rlcsd_delta: float = 0.02
    rlcsd_eta: float = 0.5
    rlcsd_two_path: bool = True

    @classmethod
    def from_config(cls, config: Any) -> SelfDistillConfig:
        node = config.get("selfdistill", None) if hasattr(config, "get") else None
        kwargs = {}
        if node is not None:
            valid = {f.name for f in fields(cls)}
            unknown = set(node.keys()) - valid
            if unknown:
                raise ValueError(f"Unknown selfdistill config keys: {sorted(unknown)}; valid: {sorted(valid)}")
            kwargs = {k: node[k] for k in node.keys()}
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"selfdistill.method must be one of {METHODS}, got {self.method!r}")
        if self.positive_source not in ("gt", "rollout"):
            raise ValueError("selfdistill.positive_source must be 'gt' or 'rollout'")
        if self.teacher_mode not in ("actor", "ref"):
            raise ValueError("selfdistill.teacher_mode must be 'actor' or 'ref'")
        if self.method == "rlcsd" and self.n_neg < 1:
            raise ValueError("RLCSD needs selfdistill.n_neg >= 1")
        if self.method in ("opsd", "rlsd") and self.teacher_mode == "ref":
            raise ValueError(
                f"{self.method} needs the student term on the same weights as the teacher; "
                "use selfdistill.teacher_mode=actor"
            )
