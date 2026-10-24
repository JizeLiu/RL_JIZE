#!/usr/bin/env bash
# Sourced by train.sbatch and the helper scripts. Establishes paths, site config and the python env.

if [[ -z "${RLX_ROOT:-}" ]]; then
  RLX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export RLX_ROOT
export VERL_ROOT="${VERL_ROOT:-$RLX_ROOT/third_party/verl}"

SITE_ENV="${SITE_ENV:-$RLX_ROOT/slurm/site.env}"
if [[ -f "$SITE_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$SITE_ENV"
else
  echo "[rlx] WARNING: $SITE_ENV not found, using slurm/site.env.example" >&2
  # shellcheck disable=SC1091
  source "$RLX_ROOT/slurm/site.env.example"
fi

if declare -F rlx_activate_env >/dev/null; then rlx_activate_env; fi

# VERL is imported from the pinned submodule (unmodified) unless it is already pip-installed.
# Our packages (rlx, rewards, data_process) resolve from the repo root.
export PYTHONPATH="$RLX_ROOT:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
