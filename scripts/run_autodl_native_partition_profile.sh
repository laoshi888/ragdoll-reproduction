#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ragdoll
export HF_HOME=/root/autodl-tmp/ragdoll/models/hf
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

CONFIG=configs/real_flex_native_partitioned.yaml

bash scripts/check_native_milvus_autodl.sh
bash scripts/start_native_milvus.sh

if [[ ! -f data/ragdoll_native_pressure.complete.json ]]; then
  python scripts/build_native_partitioned_corpus.py --config "$CONFIG" --reset
fi

python scripts/verify_native_partitioned_corpus.py --config "$CONFIG"

# The collection is released after construction, so the runtime cap can be
# lowered without retaining build-time index memory.  Equal memory and
# memory-swap values prevent the query process from escaping the RAM budget.
docker update --memory 4g --memory-swap 4g ragdoll-milvus-standalone
limit_bytes=$(docker inspect --format '{{.HostConfig.Memory}}' ragdoll-milvus-standalone)
swap_bytes=$(docker inspect --format '{{.HostConfig.MemorySwap}}' ragdoll-milvus-standalone)
echo "profile_memory_limit_bytes=$limit_bytes profile_memory_swap_bytes=$swap_bytes"

python scripts/profile_partition_residency.py \
  --config "$CONFIG" \
  --output experiments/native_partition_residency_profile.json
