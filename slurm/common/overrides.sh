#!/usr/bin/env bash
# Hydra rejects `+key=a` followed by `+key=b`. Recipes use `+` for keys VERL does not define (e.g. +selfdistill.*),
# and users may override them again on the submit command line. This keeps the *last* value per key and
# promotes to `++key=` when either occurrence used `+`, which Hydra accepts whether or not the key exists.

rlx_dedupe_overrides() {
  local -a out=()
  local -A idx=()
  local arg key prefix old okey oprefix i
  for arg in "${HYDRA_ARGS[@]}"; do
    if [[ "$arg" != *=* ]]; then out+=("$arg"); continue; fi   # e.g. ~key deletions
    key="${arg%%=*}"; prefix=""
    while [[ "$key" == [+~]* ]]; do prefix+="${key:0:1}"; key="${key:1}"; done
    if [[ -n "${idx[$key]+x}" ]]; then
      i="${idx[$key]}"; old="${out[$i]}"; okey="${old%%=*}"; oprefix=""
      while [[ "$okey" == [+~]* ]]; do oprefix+="${okey:0:1}"; okey="${okey:1}"; done
      if [[ -n "$oprefix" || -n "$prefix" ]]; then out[i]="++${key}=${arg#*=}"; else out[i]="${key}=${arg#*=}"; fi
    else
      idx[$key]=${#out[@]}
      out+=("$arg")
    fi
  done
  HYDRA_ARGS=("${out[@]}")
}
