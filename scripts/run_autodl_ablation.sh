#!/usr/bin/env bash
set -euo pipefail

# Run from the checked-out project after activating the Python 3.12 vLLM env.
PROJECT_ROOT="/root/autodl-tmp/ragdoll"
cd "$PROJECT_ROOT"

export HF_HOME="$PROJECT_ROOT/models/hf"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

python scripts/run_real_pipeline.py \
  --config configs/real_burst.yaml \
  --policies serial static adaptive
