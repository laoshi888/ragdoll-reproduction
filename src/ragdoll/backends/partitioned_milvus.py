"""Partition-resident Milvus Lite retriever used by the AutoDL reproduction."""

from __future__ import annotations

from typing import Any, Sequence

from ..contracts import RAGRequest, RetrievedRequest
from ..partitioning import (
    PartitionResidency,
    PartitionResidencySnapshot,
    partition_collection_name,
)


class PartitionedMilvusRetriever:
    """Search logical collections while retaining only a hot subset in RAM."""

    def __init__(
        self,
        *,
        uri: str,
        collection_prefix: str,
        partition_count: int,
        resident_partitions: int,
        embedder_name: str,
        top_k: int,
        client: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if client is None or embedder is None:
            try:
                from pymilvus import MilvusClient
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "PartitionedMilvusRetriever requires pymilvus and sentence-transformers; "
                    "run it on AutoDL."
                ) from error
            client = client or MilvusClient(uri=uri)
            embedder = embedder or SentenceTransformer(embedder_name)
        self._client = client
        self._embedder = embedder
        self._prefix = collection_prefix
        self._top_k = top_k
        self._residency = PartitionResidency(partition_count, resident_partitions)
        self._collections = tuple(
            partition_collection_name(collection_prefix, partition_id)
            for partition_id in range(partition_count)
        )
        missing = [name for name in self._collections if not self._client.has_collection(name)]
        if missing:
            raise ValueError(f"logical partition collections are missing: {missing}")
        for partition_id in self._residency.resident_partition_ids:
            self._load(partition_id, count=False)

    @property
    def residency_snapshot(self) -> PartitionResidencySnapshot:
        return self._residency.snapshot()

    def _load(self, partition_id: int, *, count: bool = True) -> None:
        self._client.load_collection(collection_name=self._collections[partition_id])
        if count:
            self._residency.record_cold_load()

    def _release(self, partition_id: int, *, count: bool = True) -> None:
        self._client.release_collection(collection_name=self._collections[partition_id])
        if count:
            self._residency.record_cold_release()

    @staticmethod
    def _score(hit: dict[str, Any]) -> float:
        return float(hit.get("distance", hit.get("score", float("-inf"))))

    def retrieve(self, requests: Sequence[RAGRequest]) -> Sequence[RetrievedRequest]:
        if not requests:
            return ()
        vectors = self._embedder.encode(
            [request.question for request in requests], normalize_embeddings=True
        )
        data = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        merged: list[list[tuple[float, int, str]]] = [[] for _ in requests]
        partition_scores: list[tuple[int, float]] = []

        for partition_id, collection in enumerate(self._collections):
            cold = not self._residency.is_resident(partition_id)
            if cold:
                self._load(partition_id)
            try:
                results = self._client.search(
                    collection_name=collection,
                    data=data,
                    limit=self._top_k,
                    output_fields=["text"],
                )
                self._residency.record_search()
                best = float("-inf")
                for request_index, hits in enumerate(results):
                    for hit in hits:
                        score = self._score(hit)
                        best = max(best, score)
                        merged[request_index].append(
                            (score, partition_id, hit["entity"]["text"])
                        )
                partition_scores.append((partition_id, best))
            finally:
                if cold:
                    self._release(partition_id)

        next_resident = self._residency.select_next(partition_scores)
        to_load, to_release = self._residency.apply(next_resident)
        # Release first so a policy update never temporarily exceeds the
        # configured resident collection count.
        for partition_id in to_release:
            self._release(partition_id, count=False)
        for partition_id in to_load:
            self._load(partition_id, count=False)

        return tuple(
            RetrievedRequest(
                request=request,
                contexts=tuple(
                    text
                    for _score, _partition, text in sorted(
                        candidates, key=lambda item: (item[0], -item[1]), reverse=True
                    )[: self._top_k]
                ),
            )
            for request, candidates in zip(requests, merged, strict=True)
        )

    def close(self) -> None:
        self._client.close()
