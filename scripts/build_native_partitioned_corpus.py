"""Build a scaled corpus in native Milvus Standalone partitions."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--reset", action="store_true", help="replace only the configured native collection"
    )
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--insert-batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.source_batch_size < 1 or args.insert_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    try:
        import yaml
        from pymilvus import DataType, MilvusClient
    except ImportError as error:
        raise SystemExit("Run this script in the AutoDL ragdoll environment.") from error

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_cfg = cfg["partition_source"]
    target_cfg = cfg["milvus"]
    build_cfg = cfg["native_corpus"]
    marker = Path(cfg["artifacts"]["native_corpus_marker"])
    partition_count = int(target_cfg["partition_count"])
    rows_per_partition = int(build_cfg["rows_per_partition"])
    if partition_count < 1 or rows_per_partition < 1:
        raise SystemExit("partition_count and rows_per_partition must be positive")

    source = MilvusClient(uri=source_cfg["uri"])
    target_kwargs = {"uri": target_cfg["uri"]}
    if target_cfg.get("token"):
        target_kwargs["token"] = target_cfg["token"]
    target = MilvusClient(**target_kwargs)
    collection = target_cfg["collection"]
    if target.has_collection(collection_name=collection):
        if not args.reset:
            raise SystemExit(
                f"Collection {collection!r} already exists; rerun with --reset to replace it."
            )
        target.drop_collection(collection_name=collection)
    marker.unlink(missing_ok=True)

    source_collection = source_cfg["collection"]
    source.load_collection(collection_name=source_collection)
    row_count = int(
        source.get_collection_stats(collection_name=source_collection)["row_count"]
    )
    first = source.get(
        collection_name=source_collection,
        ids=[0],
        output_fields=["vector", "text"],
    )
    if row_count < 1 or not first:
        raise SystemExit("Source collection is empty or does not contain id=0.")
    dimension = len(first[0]["vector"])

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048)
    target.create_collection(collection_name=collection, schema=schema)

    names = [
        partition_collection_name(target_cfg["partition_prefix"], partition_id)
        for partition_id in range(partition_count)
    ]
    for name in names:
        target.create_partition(collection_name=collection, partition_name=name)

    grouped: list[list[dict]] = [[] for _ in range(partition_count)]
    for start in range(0, row_count, args.source_batch_size):
        records = source.get(
            collection_name=source_collection,
            ids=list(range(start, min(start + args.source_batch_size, row_count))),
            output_fields=["vector", "text"],
        )
        for record in records:
            grouped[int(record["id"]) % partition_count].append(record)
    if any(not records for records in grouped):
        raise SystemExit("At least one native partition has no source vectors.")

    for partition_id, (name, source_records) in enumerate(zip(names, grouped, strict=True)):
        inserted = 0
        while inserted < rows_per_partition:
            stop = min(inserted + args.insert_batch_size, rows_per_partition)
            batch = []
            for offset in range(inserted, stop):
                source_record = source_records[offset % len(source_records)]
                batch.append(
                    {
                        "id": partition_id * rows_per_partition + offset,
                        "vector": source_record["vector"],
                        "text": source_record["text"],
                    }
                )
            target.insert(
                collection_name=collection,
                partition_name=name,
                data=batch,
            )
            inserted = stop
        print(
            f"partition={partition_id + 1}/{partition_count} "
            f"name={name} rows={rows_per_partition}"
        )

    target.flush(collection_name=collection)
    index_params = target.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name="vector_ivf",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": int(build_cfg["index_nlist"])},
    )
    target.create_index(
        collection_name=collection,
        index_params=index_params,
        sync=True,
    )
    target.release_collection(collection_name=collection)
    source.close()
    target.close()

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "collection": collection,
                "partitions": partition_count,
                "rows_per_partition": rows_per_partition,
                "total_rows": partition_count * rows_per_partition,
                "dimension": dimension,
                "index_type": "IVF_FLAT",
                "index_nlist": int(build_cfg["index_nlist"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"native_collection={collection} partitions={partition_count} "
        f"total={partition_count * rows_per_partition} index=IVF_FLAT complete=true"
    )


if __name__ == "__main__":
    main()
