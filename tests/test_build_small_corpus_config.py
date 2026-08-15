from pathlib import Path
import unittest


class BuildSmallCorpusConfigTests(unittest.TestCase):
    def test_builder_supports_an_optional_corpus_target(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "build_small_corpus.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn('config.get("corpus_milvus", config["milvus"])', source)


if __name__ == "__main__":
    unittest.main()
