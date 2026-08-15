#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ragdoll
export HF_HOME=/root/autodl-tmp/ragdoll/models/hf
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

CONFIG=configs/real_flex_native_smoke.yaml

bash scripts/check_native_milvus_autodl.sh
bash scripts/start_native_milvus.sh

if [[ ! -f data/ragdoll_native_smoke.complete.json ]]; then
  python scripts/build_native_partitioned_corpus.py --config "$CONFIG" --reset
fi

python scripts/verify_native_partitioned_corpus.py --config "$CONFIG"
docker update --memory 4g --memory-swap 4g ragdoll-milvus-standalone
python scripts/profile_partition_residency.py \
  --config "$CONFIG" \
  --output experiments/native_partition_smoke_profile.json
