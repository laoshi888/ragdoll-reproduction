from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.backends.native_partitioned_milvus import (  # noqa: E402
    NativePartitionedMilvusRetriever,
)
from ragdoll.contracts import RAGRequest  # noqa: E402


class FakeVectors(list):
    def tolist(self):
        return list(self)


class FakeEmbedder:
    def encode(self, questions, normalize_embeddings=True):
        return FakeVectors([[float(index)] for index, _ in enumerate(questions)])


class FakeNativeClient:
    def __init__(self):
        self.loaded = []
        self.released = []
        self.release_collection_calls = []
        self.closed = False

    def has_collection(self, *, collection_name):
        return True

    def list_partitions(self, *, collection_name):
        return ["_default", "chunks_p00", "chunks_p01", "chunks_p02"]

    def release_collection(self, *, collection_name):
        self.release_collection_calls.append(collection_name)

    def load_partitions(self, *, collection_name, partition_names):
        self.loaded.extend(partition_names)

    def release_partitions(self, *, collection_name, partition_names):
        self.released.extend(partition_names)

    def search(
        self,
        *,
        collection_name,
        partition_names,
        data,
        limit,
        output_fields,
        search_params,
    ):
        partition_id = int(partition_names[0].rsplit("p", 1)[1])
        score = {0: 0.2, 1: 0.9, 2: 0.5}[partition_id]
        return [
            [{"distance": score, "entity": {"text": f"p{partition_id}:q{index}"}}]
            for index, _ in enumerate(data)
        ]

    def close(self):
        self.closed = True


class NativePartitioningTests(unittest.TestCase):
    def test_native_partitions_merge_global_top_k_and_promote_hot_partition(self):
        client = FakeNativeClient()
        retriever = NativePartitionedMilvusRetriever(
            uri="unused",
            collection="chunks",
            partition_prefix="chunks",
            partition_count=3,
            resident_partitions=1,
            embedder_name="unused",
            top_k=2,
            client=client,
            embedder=FakeEmbedder(),
        )
        result = retriever.retrieve([RAGRequest(0, "question", 0.0)])
        self.assertEqual(result[0].contexts, ("p1:q0", "p2:q0"))
        self.assertEqual(retriever.residency_snapshot.resident_partition_ids, (1,))
        self.assertEqual(retriever.residency_snapshot.searches, 3)
        self.assertEqual(client.release_collection_calls, ["chunks"])
        self.assertEqual(client.loaded[0], "chunks_p00")
        retriever.close()
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
