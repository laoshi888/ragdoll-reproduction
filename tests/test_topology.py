from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.topology import load_topology_profiles, select_fastest_topology  # noqa: E402


class TopologySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "flex_topology_profile.json"
        cls.profiles = load_topology_profiles(path)

    def test_offload_selects_serial(self) -> None:
        selected = select_fastest_topology(self.profiles, "weight_75_gpu_25_cpu")
        self.assertEqual(selected.topology, "serial")

    def test_gpu_only_selects_adaptive(self) -> None:
        selected = select_fastest_topology(self.profiles, "gpu_only")
        self.assertEqual(selected.topology, "adaptive")

    def test_unknown_placement_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_fastest_topology(self.profiles, "unknown")


if __name__ == "__main__":
    unittest.main()
