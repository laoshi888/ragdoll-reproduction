from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.placement import PlacementProfile, select_fastest_feasible  # noqa: E402


class PlacementSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = (
            PlacementProfile("gpu_only", (100, 0, 100, 0, 100, 0), 2.7, 0.6, 50.0),
            PlacementProfile("weight_75", (75, 25, 100, 0, 100, 0), 1.8, 1.5, 21.0),
            PlacementProfile("weight_50", (50, 50, 100, 0, 100, 0), 1.6, 2.0, 16.0),
        )

    def test_uses_fastest_measured_profile_that_fits_budget(self) -> None:
        self.assertEqual(select_fastest_feasible(self.profiles, 1.9).name, "weight_75")
        self.assertEqual(select_fastest_feasible(self.profiles, 3.0).name, "gpu_only")

    def test_rejects_budget_without_a_profile(self) -> None:
        with self.assertRaises(ValueError):
            select_fastest_feasible(self.profiles, 1.5)


if __name__ == "__main__":
    unittest.main()
