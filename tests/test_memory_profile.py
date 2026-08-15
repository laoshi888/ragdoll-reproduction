from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_partition_residency.py"
SPEC = spec_from_file_location("profile_partition_residency_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MemoryProfileTests(unittest.TestCase):
    def test_parses_docker_binary_memory_units(self):
        self.assertEqual(MODULE._memory_gib("4GiB"), 4.0)
        self.assertEqual(MODULE._memory_gib("512MiB"), 0.5)

    def test_rejects_unknown_memory_units(self):
        with self.assertRaises(ValueError):
            MODULE._memory_gib("four gigabytes")


if __name__ == "__main__":
    unittest.main()
