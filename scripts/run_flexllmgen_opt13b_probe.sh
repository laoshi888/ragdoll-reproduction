#!/usr/bin/env bash
set -euo pipefail

# Minimal, reproducible FlexLLMGen probe for RAGDoll's memory-placement study.
# The six --percent values place weights, KV cache, and activations on
# GPU/CPU respectively. Default: weights 50/50, cache 100/0, activations 100/0.
PROJECT_ROOT="/root/autodl-tmp/ragdoll"
FLEX_ENV="/root/autodl-tmp/.venvs/flexllmgen"
OFFLOAD_DIR="$PROJECT_ROOT/data/flexllmgen_offload"
HF_CACHE="$PROJECT_ROOT/models/hf"
MIN_FREE_GIB=12
FLEX_PERCENT="${FLEX_PERCENT:-50 50 100 0 100 0}"

if [[ ! -x "$FLEX_ENV/bin/python" ]]; then
  echo "Missing FlexLLMGen environment: $FLEX_ENV" >&2
  exit 1
fi

available_gib=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
if (( available_gib < MIN_FREE_GIB )); then
  echo "Need at least ${MIN_FREE_GIB} GiB free; only ${available_gib} GiB available." >&2
  exit 1
fi

mkdir -p "$OFFLOAD_DIR" "$HF_CACHE"
cd "$PROJECT_ROOT"

read -r -a percent_args <<< "$FLEX_PERCENT"
if [[ ${#percent_args[@]} -ne 6 ]]; then
  echo "FLEX_PERCENT must contain six percentages; got: $FLEX_PERCENT" >&2
  exit 1
fi

export HF_HOME="$HF_CACHE"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "FlexLLMGen placement (weight, cache, activation; GPU/CPU): $FLEX_PERCENT"

"$FLEX_ENV/bin/python" -m flexllmgen.flex_opt \
  --model facebook/opt-1.3b \
  --offload-dir "$OFFLOAD_DIR" \
  --prompt-len 128 \
  --gen-len 32 \
  --gpu-batch-size 1 \
  --num-gpu-batches 1 \
  --percent "${percent_args[@]}" \
  --no-log \
  --verbose 1
