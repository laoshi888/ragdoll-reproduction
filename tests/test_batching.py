from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.batching import ProfiledBatchSelector  # noqa: E402
from ragdoll.contracts import ProfileSample, ProfileStore  # noqa: E402


class ProfiledBatchSelectorTests(unittest.TestCase):
    def test_uses_fallback_without_profiles(self) -> None:
        selector = ProfiledBatchSelector((1, 2, 4), ProfileStore(), fallback_batch_size=2)
        self.assertEqual(selector("generation", 3), 2)

    def test_selects_from_measured_candidates(self) -> None:
        profiles = ProfileStore(
            (
                ProfileSample("generation", 1, 1.0),
                ProfileSample("generation", 2, 1.2),
                ProfileSample("generation", 4, 3.0),
            )
        )
        selector = ProfiledBatchSelector((1, 2, 4), profiles)
        self.assertEqual(selector("generation", 4), 2)


if __name__ == "__main__":
    unittest.main()
