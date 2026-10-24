#!/usr/bin/env python3
"""Static check that third_party/verl still exposes every VERL internal our extensions rely on.

Pure AST / text inspection: no torch, ray or GPU needed, so it runs on a login node or in CI.
Run it after bumping the submodule; any FAIL line tells you which extension to revisit.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "third_party" / "verl"
EXPECTED_VERSION = (ROOT / "VERL_VERSION").read_text().strip().lstrip("v")

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        failures.append(msg)


def parse(rel: str) -> ast.Module:
    return ast.parse((VERL / rel).read_text())


def class_methods(tree: ast.Module, cls: str) -> dict[str, ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {n.name: n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return {}


def top_level_names(tree: ast.Module) -> set[str]:
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
    return names


def arg_names(fn: ast.FunctionDef) -> list[str]:
    return [a.arg for a in fn.args.args]


def main() -> int:
    check(VERL.exists(), f"verl source at {VERL}")
    if not VERL.exists():
        return 1

    version = (VERL / "verl/version/version").read_text().strip()
    check(version == EXPECTED_VERSION, f"verl version {version} == pinned {EXPECTED_VERSION}")

    init = (VERL / "verl/__init__.py").read_text()
    check("VERL_USE_EXTERNAL_MODULES" in init, "verl/__init__.py honours VERL_USE_EXTERNAL_MODULES (plugin loading)")

    base = parse("verl/trainer/ppo/v1/trainer_base.py")
    ppo = class_methods(base, "PPOTrainer")
    for name in ("_compute_advantage", "_compute_old_log_prob", "_compute_ref_log_prob", "_build_replay_buffer"):
        check(name in ppo, f"PPOTrainer.{name} exists")
    if "_compute_advantage" in ppo:
        check(arg_names(ppo["_compute_advantage"]) == ["self", "batch", "metrics"], "PPOTrainer._compute_advantage(self, batch, metrics)")
    check("register_trainer" in top_level_names(base), "register_trainer() exists")
    base_src = (VERL / "verl/trainer/ppo/v1/trainer_base.py").read_text()
    for attr in ("self.ref_in_actor", "self.ref_policy_wg", "self.actor_rollout_wg", "self.global_steps", "self.tokenizer"):
        check(attr in base_src, f"trainer_base uses {attr}")
    check('trainer_mode == "sync"' in base_src or 'trainer_mode != "sync"' in base_src, "trainer_mode sync gating (handled by the property override)")

    sync = parse("verl/trainer/ppo/v1/trainer_sync.py")
    check("PPOTrainerSync" in top_level_names(sync), "PPOTrainerSync exists")

    padding = parse("verl/workers/utils/padding.py")
    for fn in ("response_from_nested", "response_to_nested"):
        check(fn in top_level_names(padding), f"workers.utils.padding.{fn} exists")

    workers = class_methods(parse("verl/workers/engine_workers.py"), "ActorRolloutRefWorker")
    check("compute_log_prob" in workers and "compute_ref_log_prob" in workers, "ActorRolloutRefWorker.compute_log_prob / compute_ref_log_prob")

    core_algos = parse("verl/trainer/ppo/core_algos.py")
    for fn in ("register_adv_est", "register_policy_loss", "get_adv_estimator_fn", "get_policy_loss_fn"):
        check(fn in top_level_names(core_algos), f"core_algos.{fn} exists (algorithm extension points)")

    reward_yaml = (VERL / "verl/trainer/config/reward/reward.yaml").read_text()
    check("custom_reward_function:" in reward_yaml and "path:" in reward_yaml, "reward.custom_reward_function.{path,name} config exists")
    legacy_data = (VERL / "verl/trainer/config/data/legacy_data.yaml").read_text()
    check("custom_cls:" in legacy_data, "data.custom_cls config exists")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: re-validate the matching rlx extension before upgrading VERL.")
        return 1
    print("all compatibility checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
