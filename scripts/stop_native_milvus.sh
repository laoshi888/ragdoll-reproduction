#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ragdoll
ENV_FILE=/root/autodl-tmp/ragdoll/data/milvus-native/runtime.env
COMPOSE_FILE=configs/milvus-native-standalone-compose.yaml

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Native Milvus runtime environment does not exist; nothing to stop."
  exit 0
fi

export MILVUS_MEMORY_LIMIT=8g
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" stop
echo "Native Milvus containers stopped; persisted data was retained."
