#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ragdoll

RUNTIME_DIR=/root/autodl-tmp/ragdoll/data/milvus-native
ENV_FILE="$RUNTIME_DIR/runtime.env"
COMPOSE_FILE=configs/milvus-native-standalone-compose.yaml
mkdir -p "$RUNTIME_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  umask 077
  access_key="ragdoll$(tr -d '-' < /proc/sys/kernel/random/uuid | cut -c1-16)"
  secret_key="$(tr -d '-' < /proc/sys/kernel/random/uuid)$(tr -d '-' < /proc/sys/kernel/random/uuid)"
  printf 'MILVUS_NATIVE_DATA=%s\n' "$RUNTIME_DIR" > "$ENV_FILE"
  printf 'MILVUS_MINIO_ACCESS_KEY=%s\n' "$access_key" >> "$ENV_FILE"
  printf 'MILVUS_MINIO_SECRET_KEY=%s\n' "$secret_key" >> "$ENV_FILE"
fi

# Index construction receives extra headroom.  The experiment runners lower
# the live container to the measured 4 GiB cgroup limit before profiling.
export MILVUS_MEMORY_LIMIT=8g
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d

for _attempt in $(seq 1 36); do
  status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' ragdoll-milvus-standalone 2>/dev/null || true)
  if [[ "$status" == "healthy" ]]; then
    break
  fi
  sleep 5
done

status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' ragdoll-milvus-standalone)
if [[ "$status" != "healthy" ]]; then
  docker logs --tail 100 ragdoll-milvus-standalone
  echo "Milvus Standalone did not become healthy: $status" >&2
  exit 1
fi

limit_bytes=$(docker inspect --format '{{.HostConfig.Memory}}' ragdoll-milvus-standalone)
swap_bytes=$(docker inspect --format '{{.HostConfig.MemorySwap}}' ragdoll-milvus-standalone)
python - <<'PY'
from pymilvus import MilvusClient

client = MilvusClient(uri="http://127.0.0.1:19530")
print(f"collections={client.list_collections()}")
client.close()
PY
echo "milvus_health=$status memory_limit_bytes=$limit_bytes memory_swap_bytes=$swap_bytes"
