#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ragdoll
export HF_HOME=/root/autodl-tmp/ragdoll/models/hf
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

python scripts/verify_partitioned_corpus.py --config configs/real_flex_partitioned.yaml
python scripts/profile_joint_configuration.py \
  --config configs/real_flex_partitioned.yaml \
  --plan configs/flex_joint_profile_plan.json \
  --output experiments/flex_joint_profile.json
