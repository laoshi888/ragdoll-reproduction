#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ragdoll
export HF_HOME=/root/autodl-tmp/ragdoll/models/hf

if [[ ! -f data/ragdoll_partitioned.db.complete ]]; then
  if [[ -f data/ragdoll_partitioned.db ]]; then
    python scripts/build_partitioned_corpus.py --config configs/real_flex_partitioned.yaml --reset
  else
    python scripts/build_partitioned_corpus.py --config configs/real_flex_partitioned.yaml
  fi
fi

python scripts/verify_partitioned_corpus.py --config configs/real_flex_partitioned.yaml

python scripts/run_real_pipeline.py \
  --config configs/real_flex_partitioned.yaml \
  --policies serial \
  --output experiments/flex_partitioned_smoke.json
