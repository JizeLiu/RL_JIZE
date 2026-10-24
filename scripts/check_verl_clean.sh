#!/usr/bin/env bash
# Fail if third_party/verl differs from the commit pinned by this repo (we never patch VERL).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "$(git submodule status third_party/verl | grep -E '^[+-U]' || true)" ]]; then
  echo "third_party/verl is not at the pinned commit:" >&2
  git submodule status third_party/verl >&2
  exit 1
fi
if [[ -n "$(git -C third_party/verl status --porcelain)" ]]; then
  echo "third_party/verl has local modifications; keep changes in rlx/ instead:" >&2
  git -C third_party/verl status --short >&2
  exit 1
fi
echo "third_party/verl is clean and pinned ($(cat VERL_VERSION | head -1))"
