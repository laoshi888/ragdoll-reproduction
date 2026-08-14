#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ragdoll
export HF_HOME=/root/autodl-tmp/ragdoll/models/hf
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

python scripts/verify_partitioned_corpus.py --config configs/real_flex_partitioned.yaml
python scripts/profile_partition_residency.py \
  --config configs/real_flex_partitioned.yaml \
  --output experiments/flex_partitioned_residency_profile.json
