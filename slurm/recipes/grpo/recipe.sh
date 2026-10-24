#!/usr/bin/env bash
# GRPO, stock VERL. No rlx plugin is loaded, so this is exactly VERL's own code path.
source "$RLX_ROOT/slurm/recipes/_shared/common.sh"

HYDRA_ARGS=(
  "${COMMON_ARGS[@]}"
  algorithm.adv_estimator=grpo
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}"
)
