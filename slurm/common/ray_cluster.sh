#!/usr/bin/env bash
# Starts a Ray cluster across the Slurm allocation: head on the first node, workers on the rest.
# The training driver then runs on the head node (the batch script itself) and connects via RAY_ADDRESS.
#
# Environment inherited by Ray workers is whatever is exported *before* start_ray_cluster, which is why
# train.sbatch sources the recipe (VERL_USE_EXTERNAL_MODULES, RLX_PLUGINS, ...) first.

start_ray_cluster() {
  local port="${RAY_PORT:-6379}"
  mapfile -t RLX_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  local head="${RLX_NODES[0]}"
  local head_ip
  head_ip="$(srun --overlap --nodes=1 --ntasks=1 -w "$head" bash -c "hostname --ip-address | awk '{print \$1}'")"
  export RAY_HEAD_IP="$head_ip"
  export RAY_ADDRESS="$head_ip:$port"
  local cpus="${SLURM_CPUS_PER_TASK:-${CPUS_PER_NODE:-32}}"
  local ray_tmp="${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID}}"

  echo "[rlx] ray head on $head ($head_ip:$port), ${#RLX_NODES[@]} node(s), ${GPUS_PER_NODE} GPU(s)/node"
  srun --overlap --nodes=1 --ntasks=1 -w "$head" \
    ray start --head --node-ip-address="$head_ip" --port="$port" --temp-dir="$ray_tmp" \
    --num-cpus="$cpus" --num-gpus="$GPUS_PER_NODE" --disable-usage-stats --block &
  sleep 15

  local i
  for ((i = 1; i < ${#RLX_NODES[@]}; i++)); do
    srun --overlap --nodes=1 --ntasks=1 -w "${RLX_NODES[$i]}" \
      ray start --address="$head_ip:$port" --temp-dir="$ray_tmp" \
      --num-cpus="$cpus" --num-gpus="$GPUS_PER_NODE" --disable-usage-stats --block &
    sleep 3
  done

  local want=$(( ${#RLX_NODES[@]} * GPUS_PER_NODE )) tries=0 have=0
  while (( tries < 60 )); do
    have="$(python - <<PY 2>/dev/null || echo 0
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
print(int(ray.cluster_resources().get("GPU", 0)))
PY
)"
    [[ "$have" -ge "$want" ]] && break
    tries=$((tries + 1)); sleep 5
  done
  if [[ "$have" -lt "$want" ]]; then
    echo "[rlx] ERROR: ray sees $have/$want GPUs after waiting" >&2
    return 1
  fi
  echo "[rlx] ray cluster ready: $have GPUs"
}

stop_ray_cluster() {
  ray stop --force >/dev/null 2>&1 || true
}
