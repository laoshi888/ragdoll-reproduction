from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.partition_profile import (  # noqa: E402
    load_partition_residency_profiles,
    select_fastest_residency,
)


class PartitionProfileTests(unittest.TestCase):
    def test_selects_fastest_measured_residency(self) -> None:
        content = """{
          "summary": [
            {"resident_partitions": 2, "mean_latency_seconds": 2.3,
             "mean_retrieval_seconds": 0.1, "mean_loads": 3, "mean_releases": 3},
            {"resident_partitions": 8, "mean_latency_seconds": 2.1,
             "mean_retrieval_seconds": 0.08, "mean_loads": 0, "mean_releases": 0}
          ]
        }"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(content, encoding="utf-8")
            selected = select_fastest_residency(load_partition_residency_profiles(path), 8)
        self.assertEqual(selected.resident_partitions, 8)

    def test_rejects_duplicate_candidate(self) -> None:
        content = """{
          "summary": [
            {"resident_partitions": 2, "mean_latency_seconds": 2.3,
             "mean_retrieval_seconds": 0.1, "mean_loads": 3, "mean_releases": 3},
            {"resident_partitions": 2, "mean_latency_seconds": 2.1,
             "mean_retrieval_seconds": 0.08, "mean_loads": 0, "mean_releases": 0}
          ]
        }"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_partition_residency_profiles(path)


if __name__ == "__main__":
    unittest.main()
