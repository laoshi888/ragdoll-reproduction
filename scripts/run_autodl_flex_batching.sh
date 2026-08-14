#!/usr/bin/env bash
set -euo pipefail

# First real offloading-batch experiment: profile B=1,2 and then compare
# serial, fixed-batch, and backlog-aware scheduling on one burst workload.
PROJECT_ROOT="/root/autodl-tmp/ragdoll"
FLEX_ENV="/root/autodl-tmp/.venvs/flexllmgen"
cd "$PROJECT_ROOT"

export HF_HOME="$PROJECT_ROOT/models/hf"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

"$FLEX_ENV/bin/python" scripts/profile_backends.py \
  --config configs/real_flex_burst.yaml

"$FLEX_ENV/bin/python" scripts/run_real_pipeline.py \
  --config configs/real_flex_burst.yaml \
  --policies serial static adaptive
