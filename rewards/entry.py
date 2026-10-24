"""File passed to VERL: ``reward.custom_reward_function.path=<repo>/rewards/entry.py``.

VERL loads this file by path, so it only re-exports the dispatcher; all logic lives in the importable
``rewards`` package (the repo root must be on PYTHONPATH, which slurm/common/env.sh guarantees).
"""

from rewards.dispatch import compute_score  # noqa: F401
