#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ragdoll

echo "disk and memory capacity"
df -h /root/autodl-tmp
free -h

free_kib=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
if (( free_kib < 10 * 1024 * 1024 )); then
  echo "Need at least 10 GiB free under /root/autodl-tmp before pulling Milvus and building the native corpus." >&2
  exit 1
fi

memory_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
if (( memory_kib < 12 * 1024 * 1024 )); then
  echo "Need at least 12 GiB host memory for 8 GiB index construction plus system headroom." >&2
  exit 1
fi

echo "docker runtime"
docker version --format 'server={{.Server.Version}}'
docker compose version
docker info --format 'cgroup_driver={{.CgroupDriver}} cgroup_version={{.CgroupVersion}} storage_driver={{.Driver}}'

if docker inspect ragdoll-milvus-standalone >/dev/null 2>&1; then
  echo "existing_native_container=ragdoll-milvus-standalone"
elif command -v ss >/dev/null 2>&1 && ss -ltn | awk '{print $4}' | grep -Eq '(^|:)19530$'; then
  echo "Port 19530 is already in use by a service outside this experiment." >&2
  exit 1
fi

echo "cgroup mode"
if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
  echo "cgroup=v2"
else
  echo "cgroup=v1"
fi

echo "native_milvus_preflight=passed"
