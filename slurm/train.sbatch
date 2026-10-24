#!/usr/bin/env bash
# Generic training job. Resources are NOT hard-coded here: slurm/submit.sh passes them on the sbatch
# command line from slurm/site.env + slurm/recipes/<recipe>/resources.env.
#
# Required env (set by submit.sh): RLX_ROOT, RECIPE, RUN_DIR, OVERRIDES_FILE
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
set -euo pipefail

source "$RLX_ROOT/slurm/common/env.sh"
source "$RLX_ROOT/slurm/common/ray_cluster.sh"
source "$RLX_ROOT/slurm/common/overrides.sh"

# The recipe fills HYDRA_ARGS and may export RLX_PLUGINS; it must be sourced before Ray starts so the
# plugin variables reach every Ray worker.
source "$RLX_ROOT/slurm/recipes/$RECIPE/recipe.sh"

if [[ -n "${RLX_PLUGINS:-}" ]]; then
  export RLX_PLUGINS
  export VERL_USE_EXTERNAL_MODULES="rlx.plugin${VERL_USE_EXTERNAL_MODULES:+,$VERL_USE_EXTERNAL_MODULES}"
fi

# Per-submission overrides given on the submit.sh command line come last, so they win.
if [[ -s "${OVERRIDES_FILE:-}" ]]; then
  mapfile -t EXTRA_OVERRIDES < "$OVERRIDES_FILE"
  HYDRA_ARGS+=("${EXTRA_OVERRIDES[@]}")
fi

HYDRA_ARGS+=(
  "trainer.nnodes=$SLURM_JOB_NUM_NODES"
  "trainer.n_gpus_per_node=$GPUS_PER_NODE"
  "trainer.default_local_dir=$RUN_DIR/ckpt"
)

rlx_dedupe_overrides

trap stop_ray_cluster EXIT
start_ray_cluster

cd "$RUN_DIR"
echo "[rlx] recipe=$RECIPE nodes=$SLURM_JOB_NUM_NODES plugins=${RLX_PLUGINS:-<none>}"
printf '%s\n' "${HYDRA_ARGS[@]}" > "$RUN_DIR/hydra_args.txt"
python -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}"
