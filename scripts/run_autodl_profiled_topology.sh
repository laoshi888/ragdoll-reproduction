#!/usr/bin/env bash
set -euo pipefail

# Verify that the measured topology profile chooses serial for CPU-offloaded
# weights and adaptive for GPU-only weights under otherwise identical inputs.
PROJECT_ROOT="/root/autodl-tmp/ragdoll"
FLEX_ENV="/root/autodl-tmp/.venvs/flexllmgen"
cd "$PROJECT_ROOT"

export HF_HOME="$PROJECT_ROOT/models/hf"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

"$FLEX_ENV/bin/python" scripts/run_real_pipeline.py \
  --config configs/real_flex_burst.yaml \
  --policies profiled \
  --output "$PROJECT_ROOT/experiments/profiled_topology_offload.json"

"$FLEX_ENV/bin/python" scripts/run_real_pipeline.py \
  --config configs/real_flex_gpu_burst.yaml \
  --policies profiled \
  --output "$PROJECT_ROOT/experiments/profiled_topology_gpu.json"
