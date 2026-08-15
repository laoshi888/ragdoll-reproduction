"""Native Milvus partition residency for the Standalone reproduction."""

from __future__ import annotations

from typing import Any, Sequence

from ..contracts import RAGRequest, RetrievedRequest
from ..partitioning import (
    PartitionResidency,
    PartitionResidencySnapshot,
    partition_collection_name,
)


class NativePartitionedMilvusRetriever:
    """Search one native Milvus partition at a time under a resident-set cap.

    The persistent resident set contains exactly ``resident_partitions``.
    Searching a cold partition temporarily loads one additional partition, then
    releases it immediately.  This mirrors RAGDoll's lazy partition transfer
    between retrieval batches while preserving exact global top-k merging.
    """

    def __init__(
        self,
        *,
        uri: str,
        collection: str,
        partition_prefix: str,
        partition_count: int,
        resident_partitions: int,
        embedder_name: str,
        top_k: int,
        token: str | None = None,
        search_params: dict[str, Any] | None = None,
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
                    "NativePartitionedMilvusRetriever requires pymilvus and "
                    "sentence-transformers; run it on AutoDL."
                ) from error
            client_kwargs: dict[str, Any] = {"uri": uri}
            if token:
                client_kwargs["token"] = token
            client = client or MilvusClient(**client_kwargs)
            embedder = embedder or SentenceTransformer(embedder_name)
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._top_k = top_k
        self._search_params = search_params or {"metric_type": "COSINE", "params": {}}
        self._residency = PartitionResidency(partition_count, resident_partitions)
        self._partitions = tuple(
            partition_collection_name(partition_prefix, partition_id)
            for partition_id in range(partition_count)
        )
        if not self._client.has_collection(collection_name=collection):
            raise ValueError(f"native Milvus collection is missing: {collection}")
        present = set(self._client.list_partitions(collection_name=collection))
        missing = [name for name in self._partitions if name not in present]
        if missing:
            raise ValueError(f"native Milvus partitions are missing: {missing}")

        # Standalone persists loaded state.  Each isolated profile must start
        # from the configured resident set rather than inherit a previous run.
        self._client.release_collection(collection_name=self._collection)
        for partition_id in self._residency.resident_partition_ids:
            self._load(partition_id, count=False)

    @property
    def residency_snapshot(self) -> PartitionResidencySnapshot:
        return self._residency.snapshot()

    def _load(self, partition_id: int, *, count: bool = True) -> None:
        self._client.load_partitions(
            collection_name=self._collection,
            partition_names=[self._partitions[partition_id]],
        )
        if count:
            self._residency.record_cold_load()

    def _release(self, partition_id: int, *, count: bool = True) -> None:
        self._client.release_partitions(
            collection_name=self._collection,
            partition_names=[self._partitions[partition_id]],
        )
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

        for partition_id, partition_name in enumerate(self._partitions):
            cold = not self._residency.is_resident(partition_id)
            if cold:
                self._load(partition_id)
            try:
                results = self._client.search(
                    collection_name=self._collection,
                    partition_names=[partition_name],
                    data=data,
                    limit=self._top_k,
                    output_fields=["text"],
                    search_params=self._search_params,
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
