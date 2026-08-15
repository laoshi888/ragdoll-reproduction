"""Verify native Milvus partition names, row counts, and index metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragdoll.partitioning import partition_collection_name  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_flex_native_partitioned.yaml",
    )
    args = parser.parse_args()
    try:
        import yaml
        from pymilvus import MilvusClient
    except ImportError as error:
        raise SystemExit("Run this script in the AutoDL ragdoll environment.") from error

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    milvus = cfg["milvus"]
    client_kwargs = {"uri": milvus["uri"]}
    if milvus.get("token"):
        client_kwargs["token"] = milvus["token"]
    client = MilvusClient(**client_kwargs)
    collection = milvus["collection"]
    if not client.has_collection(collection_name=collection):
        raise SystemExit(f"Missing native collection: {collection}")
    present = set(client.list_partitions(collection_name=collection))
    counts = []
    for partition_id in range(int(milvus["partition_count"])):
        name = partition_collection_name(milvus["partition_prefix"], partition_id)
        if name not in present:
            raise SystemExit(f"Missing native partition: {name}")
        counts.append(
            int(
                client.get_partition_stats(
                    collection_name=collection, partition_name=name
                )["row_count"]
            )
        )
    indexes = client.list_indexes(collection_name=collection)
    client.close()
    expected_per_partition = int(cfg["native_corpus"]["rows_per_partition"])
    if any(count != expected_per_partition for count in counts):
        raise SystemExit(
            f"Native partition counts differ from expected {expected_per_partition}: {counts}"
        )
    if "vector_ivf" not in indexes:
        raise SystemExit(f"Expected vector_ivf index, found: {indexes}")
    print(
        f"native_partitions={len(counts)} rows_per_partition={expected_per_partition} "
        f"total={sum(counts)} indexes={indexes} verified=true"
    )


if __name__ == "__main__":
    main()
