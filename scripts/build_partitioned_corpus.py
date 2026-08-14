"""Copy the existing small corpus into Milvus-Lite logical partitions."""

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
    parser.add_argument(
        "--reset", action="store_true", help="replace only the configured logical collections"
    )
    parser.add_argument("--copy-batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.copy_batch_size < 1:
        raise SystemExit("--copy-batch-size must be positive")
    try:
        import yaml
        from pymilvus import MilvusClient
    except ImportError as error:
        raise SystemExit("Run this script in the AutoDL ragdoll environment.") from error

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_cfg = cfg["partition_source"]
    target_cfg = cfg["milvus"]
    source = MilvusClient(source_cfg["uri"])
    target = MilvusClient(target_cfg["uri"])
    completion_marker = Path(f"{target_cfg['uri']}.complete")
    prefix = target_cfg["collection_prefix"]
    partition_count = int(target_cfg["partition_count"])
    names = [partition_collection_name(prefix, item) for item in range(partition_count)]
    existing = [name for name in names if target.has_collection(name)]
    if existing and not args.reset:
        raise SystemExit(
            f"Logical collections already exist: {existing}; use --reset to replace only these collections."
        )
    for name in existing:
        target.drop_collection(name)
    completion_marker.unlink(missing_ok=True)

    row_count = int(
        source.get_collection_stats(collection_name=source_cfg["collection"])["row_count"]
    )
    if row_count < 1:
        raise SystemExit("Source collection is empty.")
    # Persisted Milvus Lite collections are commonly reopened in the released
    # state.  Explicit loading is required before get/query/search operations.
    source.load_collection(collection_name=source_cfg["collection"])
    first = source.get(
        collection_name=source_cfg["collection"], ids=[0], output_fields=["vector", "text"]
    )
    if not first:
        raise SystemExit("Source collection does not contain id=0.")
    dimension = len(first[0]["vector"])
    for name in names:
        target.create_collection(
            collection_name=name,
            dimension=dimension,
            metric_type="COSINE",
            auto_id=False,
        )

    counts = [0] * partition_count
    for start in range(0, row_count, args.copy_batch_size):
        records = source.get(
            collection_name=source_cfg["collection"],
            ids=list(range(start, min(start + args.copy_batch_size, row_count))),
            output_fields=["vector", "text"],
        )
        grouped: list[list[dict]] = [[] for _ in range(partition_count)]
        for record in records:
            partition_id = int(record["id"]) % partition_count
            grouped[partition_id].append(
                {"id": int(record["id"]), "vector": record["vector"], "text": record["text"]}
            )
        for partition_id, batch in enumerate(grouped):
            if batch:
                target.insert(collection_name=names[partition_id], data=batch)
                counts[partition_id] += len(batch)
        print(f"copied={min(start + args.copy_batch_size, row_count)}/{row_count}")

    source.close()
    target.close()
    completion_marker.write_text(
        f"partitions={partition_count}\ntotal={sum(counts)}\n", encoding="utf-8"
    )
    print(f"logical_partitions={partition_count} counts={counts} total={sum(counts)}")


if __name__ == "__main__":
    main()
