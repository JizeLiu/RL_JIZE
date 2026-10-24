#!/usr/bin/env bash
# PPO with a learned critic (GAE), stock VERL. No rlx plugin.
source "$RLX_ROOT/slurm/recipes/_shared/common.sh"

CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-$MODEL_PATH}"
CRITIC_LR="${CRITIC_LR:-1e-5}"

HYDRA_ARGS=(
  "${COMMON_ARGS[@]}"
  algorithm.adv_estimator=gae
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}"
  # PPO needs one rollout per prompt unless you deliberately want a larger group
  actor_rollout_ref.rollout.n="${ROLLOUT_N_PPO:-1}"
  critic.model.path="${CRITIC_MODEL_PATH}"
  critic.model.use_remove_padding=True
  critic.model.enable_gradient_checkpointing=True
  critic.optim.lr="${CRITIC_LR}"
  critic.use_dynamic_bsz=True
  critic.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  critic.fsdp.param_offload=False
  critic.fsdp.optimizer_offload=False
  trainer.critic_warmup=0
)
