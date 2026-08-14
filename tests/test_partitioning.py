from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.backends.partitioned_milvus import PartitionedMilvusRetriever  # noqa: E402
from ragdoll.contracts import RAGRequest  # noqa: E402
from ragdoll.partitioning import PartitionResidency, partition_collection_name  # noqa: E402


class FakeVectors(list):
    def tolist(self):
        return list(self)


class FakeEmbedder:
    def encode(self, questions, normalize_embeddings=True):
        return FakeVectors([[float(index)] for index, _ in enumerate(questions)])


class FakeClient:
    def __init__(self):
        self.loaded = []
        self.released = []
        self.closed = False

    def has_collection(self, name):
        return True

    def load_collection(self, *, collection_name):
        self.loaded.append(collection_name)

    def release_collection(self, *, collection_name):
        self.released.append(collection_name)

    def search(self, *, collection_name, data, limit, output_fields):
        partition_id = int(collection_name.rsplit("p", 1)[1])
        score = {0: 0.2, 1: 0.9, 2: 0.5}[partition_id]
        return [
            [{"distance": score, "entity": {"text": f"p{partition_id}:q{index}"}}]
            for index, _ in enumerate(data)
        ]

    def close(self):
        self.closed = True


class PartitioningTests(unittest.TestCase):
    def test_collection_names_are_stable(self):
        self.assertEqual(partition_collection_name("chunks", 3), "chunks_p03")

    def test_hot_selection_keeps_existing_resident_on_tie(self):
        residency = PartitionResidency(partition_count=3, resident_count=1)
        self.assertEqual(residency.select_next([(0, 0.7), (1, 0.7), (2, 0.2)]), (0,))

    def test_retriever_merges_global_top_k_and_promotes_hot_partition(self):
        client = FakeClient()
        retriever = PartitionedMilvusRetriever(
            uri="unused",
            collection_prefix="chunks",
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
        retriever.close()
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
