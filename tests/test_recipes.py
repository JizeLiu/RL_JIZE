"""Compose every Slurm recipe against VERL's real Hydra config and hand the result to our own config parser."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("hydra")

import validate_recipes as vr  # noqa: E402

from rlx.selfdistill.config import SelfDistillConfig  # noqa: E402

RECIPES = sorted(p.name for p in (ROOT / "slurm" / "recipes").iterdir() if p.is_dir() and not p.name.startswith("_"))


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_composes_against_verl_config(recipe):
    cfg = vr.compose_recipe(recipe)
    assert cfg.trainer.nnodes == 2 and cfg.reward.custom_reward_function.name == "compute_score"
    assert cfg.reward.custom_reward_function.path.endswith("rewards/entry.py")


@pytest.mark.parametrize("recipe,method", [("rlsd", "rlsd"), ("opsd", "opsd"), ("rlcsd", "rlcsd")])
def test_selfdistill_recipes_parse(recipe, method):
    cfg = vr.compose_recipe(recipe)
    assert cfg.trainer.v1.trainer_mode == "selfdistill_sync"
    sd = SelfDistillConfig.from_config(cfg)
    assert sd.method == method


def test_user_override_of_plus_key_wins():
    cfg = vr.compose_recipe("rlcsd", ["+selfdistill.n_neg=2", "actor_rollout_ref.rollout.n=4"])
    assert SelfDistillConfig.from_config(cfg).n_neg == 2
    assert cfg.actor_rollout_ref.rollout.n == 4


def test_stock_recipes_do_not_touch_selfdistill_or_plugins():
    for r in ("grpo", "ppo", "justrl2"):
        cfg = vr.compose_recipe(r)
        assert cfg.trainer.v1.trainer_mode == "sync"
        assert cfg.get("selfdistill") is None


def test_justrl_matches_paper_table():
    cfg = vr.compose_recipe("justrl2")
    a = cfg.actor_rollout_ref
    assert cfg.algorithm.adv_estimator == "grpo"
    assert cfg.data.train_batch_size == 256 and a.actor.ppo_mini_batch_size == 64
    assert a.rollout.n == 8 and a.rollout.temperature == 1.0
    assert (a.actor.clip_ratio_low, a.actor.clip_ratio_high) == (0.2, 0.28)
    assert a.actor.use_kl_loss is False and a.actor.entropy_coeff == 0
    assert a.actor.optim.lr == 1e-6 and a.actor.ppo_micro_batch_size_per_gpu == 1
    assert cfg.data.max_response_length + cfg.data.max_prompt_length == 16384


def test_unknown_selfdistill_key_is_rejected():
    cfg = vr.compose_recipe("rlsd", ["+selfdistill.typo_key=1"])
    with pytest.raises(ValueError, match="Unknown selfdistill"):
        SelfDistillConfig.from_config(cfg)
