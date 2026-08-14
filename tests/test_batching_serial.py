from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.batching import ProfiledSerialBatchSelector  # noqa: E402
from ragdoll.contracts import ProfileSample, ProfileStore  # noqa: E402


class ProfiledSerialBatchSelectorTests(unittest.TestCase):
    def test_combines_both_stage_costs(self) -> None:
        profiles = ProfileStore(
            (
                ProfileSample("retrieval", 1, 1.0),
                ProfileSample("generation", 1, 3.0),
                ProfileSample("retrieval", 2, 1.0),
                ProfileSample("generation", 2, 3.0),
            )
        )
        selector = ProfiledSerialBatchSelector((1, 2), profiles)
        self.assertEqual(selector("serial", 4), 2)


if __name__ == "__main__":
    unittest.main()
