#!/usr/bin/env bash
set -euo pipefail

# GPU-only counterpart of the offload batching experiment.  Results isolate
# whether offload's CPU traffic, rather than batching alone, causes contention.
PROJECT_ROOT="/root/autodl-tmp/ragdoll"
FLEX_ENV="/root/autodl-tmp/.venvs/flexllmgen"
cd "$PROJECT_ROOT"

export HF_HOME="$PROJECT_ROOT/models/hf"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

"$FLEX_ENV/bin/python" scripts/profile_backends.py \
  --config configs/real_flex_gpu_burst.yaml

RESULT_DIR="$PROJECT_ROOT/experiments/flex_gpu_batching_runs"
mkdir -p "$RESULT_DIR"

orders=(
  "serial static adaptive"
  "static adaptive serial"
  "adaptive serial static"
)
round=1
for order in "${orders[@]}"; do
  for policy in $order; do
    "$FLEX_ENV/bin/python" scripts/run_real_pipeline.py \
      --config configs/real_flex_gpu_burst.yaml \
      --policies "$policy" \
      --output "$RESULT_DIR/round${round}_${policy}.json"
  done
  round=$((round + 1))
done

"$FLEX_ENV/bin/python" scripts/summarize_policy_runs.py \
  --input-dir "$RESULT_DIR" \
  --output "$PROJECT_ROOT/experiments/flex_gpu_batching_summary.json"
