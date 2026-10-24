#!/usr/bin/env bash
# OPSD as a sampled-token reverse-KL advantage: A_t = clip(log P_T - log P_S), clipped pointwise.
# The verifier reward is not used for the update; the dataset reference hint is the only signal source. Cost: +1 forward per rollout.
source "$RLX_ROOT/slurm/recipes/_shared/selfdistill.sh"

HYDRA_ARGS=(
  "${SELFDISTILL_BASE_ARGS[@]}"
  actor_rollout_ref.actor.use_kl_loss=False
  +selfdistill.method=opsd
  +selfdistill.teacher_mode=actor
  "+selfdistill.opsd_clip=${SD_OPSD_CLIP:-5.0}"
)
