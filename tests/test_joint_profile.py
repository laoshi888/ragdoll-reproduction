from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.joint_profile import (  # noqa: E402
    load_joint_configuration_profiles,
    select_fastest_joint_configuration,
)


class JointProfileTests(unittest.TestCase):
    def _profiles(self):
        content = """{
          "summary": [
            {"name": "offload", "max_gpu_memory_gib": 1.8,
             "placement": "weight_75_gpu_25_cpu", "resident_partitions": 8,
             "topology": "serial", "median_latency_seconds": 2.2,
             "mean_latency_seconds": 2.6},
            {"name": "gpu", "max_gpu_memory_gib": 3.0,
             "placement": "gpu_only", "resident_partitions": 8,
             "topology": "adaptive", "median_latency_seconds": 1.3,
             "mean_latency_seconds": 2.1}
          ]
        }"""
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "joint.json"
        path.write_text(content, encoding="utf-8")
        return directory, load_joint_configuration_profiles(path)

    def test_budget_selects_measured_offload_configuration(self) -> None:
        directory, profiles = self._profiles()
        self.addCleanup(directory.cleanup)
        self.assertEqual(select_fastest_joint_configuration(profiles, 1.8).name, "offload")

    def test_larger_budget_selects_gpu_only_by_median_latency(self) -> None:
        directory, profiles = self._profiles()
        self.addCleanup(directory.cleanup)
        self.assertEqual(select_fastest_joint_configuration(profiles, 3.0).name, "gpu")


if __name__ == "__main__":
    unittest.main()
