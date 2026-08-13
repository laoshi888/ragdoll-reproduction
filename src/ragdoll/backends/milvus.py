"""Milvus + Sentence Transformers retriever, loaded only on AutoDL runtime."""

from __future__ import annotations

from typing import Sequence

from ..contracts import RAGRequest, RetrievedRequest


class MilvusRetriever:
    """Embed questions and retrieve stored text chunks from a Milvus collection."""

    def __init__(self, *, uri: str, collection: str, embedder_name: str, top_k: int) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        try:
            from pymilvus import MilvusClient
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "MilvusRetriever requires pymilvus and sentence-transformers; run it on AutoDL."
            ) from error
        self._client = MilvusClient(uri=uri)
        self._collection = collection
        # Milvus Lite can leave a persisted collection in the ``released``
        # state after the process that built it exits.  Searching a released
        # collection fails, so make the runtime ownership explicit here.
        self._client.load_collection(collection_name=self._collection)
        self._embedder = SentenceTransformer(embedder_name)
        self._top_k = top_k

    def retrieve(self, requests: Sequence[RAGRequest]) -> Sequence[RetrievedRequest]:
        if not requests:
            return ()
        vectors = self._embedder.encode([request.question for request in requests], normalize_embeddings=True)
        results = self._client.search(
            collection_name=self._collection,
            data=vectors.tolist(),
            limit=self._top_k,
            output_fields=["text"],
        )
        return tuple(
            RetrievedRequest(
                request=request,
                contexts=tuple(hit["entity"]["text"] for hit in hits),
            )
            for request, hits in zip(requests, results, strict=True)
        )
