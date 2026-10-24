#!/usr/bin/env bash
# Algorithm-agnostic VERL overrides. Recipes source this and append their algorithm-specific args.
# Every knob is an env var so `FOO=bar slurm/submit.sh grpo` works; Hydra overrides on the submit
# command line still win over everything here.

: "${MODEL_PATH:?set MODEL_PATH in slurm/site.env}"
: "${TRAIN_FILES:?set TRAIN_FILES in slurm/site.env}"
: "${VAL_FILES:?set VAL_FILES in slurm/site.env}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
PROJECT_NAME="${PROJECT_NAME:-rlx}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RECIPE:-run}_$(date +%Y%m%d_%H%M%S)}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
REWARD_FILE="${REWARD_FILE:-$RLX_ROOT/rewards/entry.py}"

COMMON_ARGS=(
  algorithm.use_kl_in_reward=False

  "data.train_files=${TRAIN_FILES}"
  "data.val_files=${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.truncation=error

  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True

  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

  actor_rollout_ref.rollout.name="${INFER_BACKEND}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"

  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True

  reward.custom_reward_function.path="${REWARD_FILE}"
  reward.custom_reward_function.name=compute_score

  trainer.balance_batch=True
  "trainer.logger=${TRAINER_LOGGER}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.total_epochs="${TOTAL_EPOCHS}"

  # We run the python env we activated, not a per-worker `uv run` environment.
  ray_kwargs.ray_init.runtime_env.py_executable=null
)
