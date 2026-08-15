from pathlib import Path
import unittest


class NativeCorpusConfigTests(unittest.TestCase):
    def test_native_pressure_scale_and_memory_budget_are_consistent(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "real_flex_native_partitioned.yaml"
        )
        content = path.read_text(encoding="utf-8")
        self.assertIn("mode: native_partitions", content)
        self.assertIn("partition_count: 32", content)
        self.assertIn("rows_per_partition: 32768", content)
        self.assertIn("host_budget_gib: 8.0", content)
        self.assertIn("llm_host_reserve_gib: 4.0", content)
        self.assertIn("milvus_container_limit_gib: 4.0", content)


if __name__ == "__main__":
    unittest.main()
