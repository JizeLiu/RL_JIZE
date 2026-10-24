#!/usr/bin/env bash
# RLCSD (arXiv 2606.11709): contrast the teacher under a correct hint with K wrong-hint teachers.
# Cost: +(1+K) forward passes per rollout. Paper defaults: K=4, tau=1.3, lambda=0.5, delta=0.02, eta=0.5,
# frozen (reference) teacher, which needs a ref model, hence use_kl_loss=True with a zero coefficient.
source "$RLX_ROOT/slurm/recipes/_shared/selfdistill.sh"

HYDRA_ARGS=(
  "${SELFDISTILL_BASE_ARGS[@]}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  +selfdistill.method=rlcsd
  "+selfdistill.teacher_mode=${SD_TEACHER_MODE:-ref}"
  "+selfdistill.n_neg=${SD_N_NEG:-4}"
  "+selfdistill.rlcsd_tau=${SD_TAU:-1.3}"
  "+selfdistill.rlcsd_lam=${SD_LAM:-0.5}"
  "+selfdistill.rlcsd_delta=${SD_DELTA:-0.02}"
  "+selfdistill.rlcsd_eta=${SD_ETA:-0.5}"
)
