"""Configuration-driven retriever construction."""

from __future__ import annotations

from .backends.milvus import MilvusRetriever


def build_retriever(cfg: dict):
    milvus = cfg["milvus"]
    mode = milvus.get("mode", "single")
    common = {
        "embedder_name": cfg["models"]["embedder"],
        "top_k": cfg["run"]["top_k"],
    }
    if mode == "single":
        return MilvusRetriever(
            uri=milvus["uri"], collection=milvus["collection"], **common
        )
    if mode == "logical_partitions":
        from .backends.partitioned_milvus import PartitionedMilvusRetriever

        return PartitionedMilvusRetriever(
            uri=milvus["uri"],
            collection_prefix=milvus["collection_prefix"],
            partition_count=milvus["partition_count"],
            resident_partitions=milvus["resident_partitions"],
            **common,
        )
    raise ValueError(f"unsupported Milvus mode: {mode}")
