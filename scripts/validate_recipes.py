#!/usr/bin/env python3
"""Compose every Slurm recipe against VERL's real Hydra config (strict struct mode) so typos in override
keys fail here, not after a 2-hour queue wait. Needs only hydra-core + omegaconf (no GPU, no Ray).

    python scripts/validate_recipes.py            # all recipes
    python scripts/validate_recipes.py rlsd grpo  # a subset
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
VERL_CONFIG = ROOT / "third_party" / "verl" / "verl" / "trainer" / "config"
SCHEDULER_ARGS = ["trainer.nnodes=2", "trainer.n_gpus_per_node=8", "trainer.default_local_dir=/tmp/rlx_validate/ckpt"]


def recipe_args(recipe: str, extra: list[str] | None = None) -> list[str]:
    env = {
        **os.environ,
        "RLX_ROOT": str(ROOT),
        "RECIPE": recipe,
        "MODEL_PATH": "/models/dummy",
        "TRAIN_FILES": "['/data/train.parquet']",
        "VAL_FILES": "['/data/val.parquet']",
    }
    script = (
        f'source "{ROOT}/slurm/common/overrides.sh"; source "{ROOT}/slurm/recipes/{recipe}/recipe.sh" && '
        'HYDRA_ARGS+=("$@"); rlx_dedupe_overrides; printf "%s\\n" "${HYDRA_ARGS[@]}"'
    )
    out = subprocess.run(["bash", "-c", script, "_", *(extra or [])], env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"recipe {recipe} failed to source:\n{out.stderr}")
    return [line for line in out.stdout.splitlines() if line]


def compose_recipe(recipe: str, extra: list[str] | None = None):
    overrides = recipe_args(recipe, extra) + SCHEDULER_ARGS
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(VERL_CONFIG), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def validate(recipe: str) -> None:
    cfg = compose_recipe(recipe)
    algo = cfg.algorithm.adv_estimator
    mode = cfg.trainer.v1.trainer_mode
    sd = cfg.get("selfdistill")
    print(f"ok   {recipe:8s} adv={algo:5s} trainer_mode={mode:16s} selfdistill={dict(sd) if sd else None}")


def main() -> int:
    recipes = sys.argv[1:] or sorted(p.name for p in (ROOT / "slurm" / "recipes").iterdir() if p.is_dir() and not p.name.startswith("_"))
    failed = 0
    for r in recipes:
        try:
            validate(r)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {r}: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
