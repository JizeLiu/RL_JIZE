#!/usr/bin/env bash
# JustRL-style minimal recipe (arXiv 2512.16649, Table 2): plain GRPO, binary outcome reward, no KL,
# no entropy bonus, constant LR, one technique only ("clip higher"), single stage.
#
# Everything below is a VERL config override, no custom code. The paper's numbers are the defaults;
# set the env vars to deviate. For the paper's exact reward use data_source=math_dapo in the dataset
# (falls through rewards/routes.yaml to VERL's built-in DAPO verifier).
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-15360}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
source "$RLX_ROOT/slurm/recipes/_shared/common.sh"

HYDRA_ARGS=(
  "${COMMON_ARGS[@]}"
  algorithm.adv_estimator=grpo
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW:-0.2}"
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH:-0.28}"
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
)
