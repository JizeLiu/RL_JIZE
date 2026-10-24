#!/usr/bin/env bash
# RLSD (arXiv 2604.03128): verifier advantage sets the direction, the privileged self-teacher modulates
# per-token magnitude. Cost: +1 forward pass per rollout.
source "$RLX_ROOT/slurm/recipes/_shared/selfdistill.sh"

HYDRA_ARGS=(
  "${SELFDISTILL_BASE_ARGS[@]}"
  actor_rollout_ref.actor.use_kl_loss=False
  +selfdistill.method=rlsd
  +selfdistill.teacher_mode=actor
  "+selfdistill.eps_w=${SD_EPS_W:-0.2}"
  "+selfdistill.lam_start=${SD_LAM_START:-0.5}"
  "+selfdistill.lam_end=${SD_LAM_END:-0.0}"
  "+selfdistill.lam_decay_steps=${SD_LAM_DECAY_STEPS:-50}"
)
