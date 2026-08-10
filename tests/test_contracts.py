from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.contracts import ProfileSample, ProfileStore  # noqa: E402


class ProfileStoreTests(unittest.TestCase):
    def test_round_trip_and_mean(self) -> None:
        store = ProfileStore()
        store.add(ProfileSample(stage="generation", batch_size=4, elapsed_seconds=2.0))
        store.add(ProfileSample(stage="generation", batch_size=4, elapsed_seconds=4.0))
        self.assertEqual(store.mean_seconds("generation", 4), 3.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile.json"
            store.save(path)
            loaded = ProfileStore.load(path)
        self.assertEqual(loaded.samples, store.samples)

    def test_invalid_profile_sample_is_rejected(self) -> None:
        store = ProfileStore()
        with self.assertRaises(ValueError):
            store.add(ProfileSample(stage="retrieval", batch_size=0, elapsed_seconds=1.0))


if __name__ == "__main__":
    unittest.main()
