#!/usr/bin/env bash
# Shared by the OPSD / RLSD / RLCSD recipes: GRPO + the rlx self-distillation trainer.
#
#  * trainer.v1.trainer_mode=selfdistill_sync selects rlx.selfdistill.trainer (registered through the
#    VERL plugin hook, so VERL source is untouched);
#  * the +selfdistill.* keys are read only by that trainer;
#  * every dataset row needs reward_model.ground_truth (and ideally extra_info.solution) for GT hints,
#    see data_process/SCHEMA.md.
source "$RLX_ROOT/slurm/recipes/_shared/common.sh"

export RLX_PLUGINS="rlx.selfdistill.trainer"

SELFDISTILL_BASE_ARGS=(
  "${COMMON_ARGS[@]}"
  algorithm.adv_estimator=grpo
  trainer.v1.trainer_mode=selfdistill_sync
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}"
  "+selfdistill.positive_source=${SD_POSITIVE_SOURCE:-gt}"
  "+selfdistill.seed=${SD_SEED:-0}"
)
