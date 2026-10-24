#!/usr/bin/env bash
# Submit a training recipe to Slurm.
#
#   slurm/submit.sh <recipe> [--dry-run] [--nodes N] [--time HH:MM:SS] [--name NAME] [hydra.key=value ...]
#
# Examples:
#   slurm/submit.sh grpo
#   slurm/submit.sh rlsd --nodes 4 actor_rollout_ref.rollout.n=8 +selfdistill.lam_start=0.3
#   slurm/submit.sh justrl2 --dry-run
#
# Anything that looks like key=value is forwarded to VERL's Hydra config and wins over the recipe.
set -euo pipefail

RLX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RLX_ROOT
source "$RLX_ROOT/slurm/common/env.sh"

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}"; echo; echo "recipes: $(ls "$RLX_ROOT/slurm/recipes" | grep -v '^_' | tr '\n' ' ')"; }

[[ $# -ge 1 ]] || { usage; exit 1; }
RECIPE="$1"; shift
[[ -f "$RLX_ROOT/slurm/recipes/$RECIPE/recipe.sh" ]] || { echo "unknown recipe: $RECIPE" >&2; usage; exit 1; }

# Defaults, then recipe resources, then CLI flags.
NODES=1; TIME="48:00:00"; JOB_NAME="$RECIPE"; DRY_RUN=0
# shellcheck disable=SC1090
source "$RLX_ROOT/slurm/recipes/$RECIPE/resources.env"

OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --nodes) NODES="$2"; shift ;;
    --time) TIME="$2"; shift ;;
    --name) JOB_NAME="$2"; shift ;;
    *=*) OVERRIDES+=("$1") ;;
    *) echo "unrecognised argument: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

STAMP="$(date +%Y%m%d_%H%M%S)"
export RECIPE
export RUN_DIR="${RUN_ROOT:-$RLX_ROOT/runs}/${JOB_NAME}_${STAMP}"
export OVERRIDES_FILE="$RUN_DIR/overrides.txt"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${JOB_NAME}_${STAMP}}"

SBATCH_ARGS=(
  --job-name="$JOB_NAME" --nodes="$NODES" --time="$TIME"
  --gres="gpu:$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_NODE"
  --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err"
)
[[ -n "${SLURM_PARTITION:-}" ]] && SBATCH_ARGS+=(--partition="$SLURM_PARTITION")
[[ -n "${SLURM_ACCOUNT:-}" ]] && SBATCH_ARGS+=(--account="$SLURM_ACCOUNT")
[[ -n "${SLURM_QOS:-}" ]] && SBATCH_ARGS+=(--qos="$SLURM_QOS")
[[ -n "${MEM_PER_NODE:-}" ]] && SBATCH_ARGS+=(--mem="$MEM_PER_NODE")

if [[ "$DRY_RUN" == 1 ]]; then
  echo "RUN_DIR=$RUN_DIR"
  echo "sbatch ${SBATCH_ARGS[*]} $RLX_ROOT/slurm/train.sbatch"
  echo "overrides: ${OVERRIDES[*]:-<none>}"
  exit 0
fi

mkdir -p "$RUN_DIR"
printf '%s\n' "${OVERRIDES[@]:-}" | sed '/^$/d' > "$OVERRIDES_FILE"
{
  echo "recipe: $RECIPE"
  echo "submitted: $STAMP"
  echo "rlx_commit: $(git -C "$RLX_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "rlx_dirty: $(git -C "$RLX_ROOT" status --porcelain 2>/dev/null | wc -l)"
  echo "verl_commit: $(git -C "$VERL_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "verl_dirty: $(git -C "$VERL_ROOT" status --porcelain 2>/dev/null | wc -l)"
} > "$RUN_DIR/meta.txt"
cp -r "$RLX_ROOT/slurm/recipes/$RECIPE" "$RUN_DIR/recipe_snapshot"

sbatch "${SBATCH_ARGS[@]}" "$RLX_ROOT/slurm/train.sbatch"
