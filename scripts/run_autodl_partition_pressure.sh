#!/usr/bin/env bash
set -euo pipefail

# First run the preflight manually and confirm sufficient remote disk space.
# This script creates only the pressure experiment's configured database files.
cd /root/autodl-tmp/ragdoll
export HF_HOME=/root/autodl-tmp/ragdoll/models/hf
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

CONFIG=configs/real_flex_partition_pressure.yaml
if [[ ! -f data/ragdoll_pressure_source.db ]]; then
  python scripts/build_small_corpus.py --config "$CONFIG"
fi
if [[ ! -f data/ragdoll_pressure_partitioned.db.complete ]]; then
  if [[ -f data/ragdoll_pressure_partitioned.db ]]; then
    python scripts/build_partitioned_corpus.py --config "$CONFIG" --reset
  else
    python scripts/build_partitioned_corpus.py --config "$CONFIG"
  fi
fi

python scripts/verify_partitioned_corpus.py --config "$CONFIG"
python scripts/profile_partition_residency.py \
  --config "$CONFIG" \
  --output experiments/flex_partition_pressure_profile.json
