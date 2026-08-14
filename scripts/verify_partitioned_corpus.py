"""Verify logical partition counts without loading an embedding model."""

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
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "real_flex_partitioned.yaml"
    )
    args = parser.parse_args()
    try:
        import yaml
        from pymilvus import MilvusClient
    except ImportError as error:
        raise SystemExit("Run this script in the AutoDL ragdoll environment.") from error

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    milvus = cfg["milvus"]
    client = MilvusClient(milvus["uri"])
    counts: list[int] = []
    for partition_id in range(int(milvus["partition_count"])):
        name = partition_collection_name(milvus["collection_prefix"], partition_id)
        if not client.has_collection(name):
            raise SystemExit(f"Missing logical partition collection: {name}")
        counts.append(int(client.get_collection_stats(collection_name=name)["row_count"]))
    client.close()
    source = MilvusClient(cfg["partition_source"]["uri"])
    source.load_collection(collection_name=cfg["partition_source"]["collection"])
    expected = int(
        source.get_collection_stats(
            collection_name=cfg["partition_source"]["collection"]
        )["row_count"]
    )
    source.close()
    if sum(counts) != expected:
        raise SystemExit(f"Partition total {sum(counts)} does not match source total {expected}")
    print(f"logical_partitions={len(counts)} counts={counts} total={sum(counts)} verified=true")


if __name__ == "__main__":
    main()
