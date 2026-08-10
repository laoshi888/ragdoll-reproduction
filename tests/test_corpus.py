from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.corpus import chunk_text, record_contexts  # noqa: E402


class CorpusTests(unittest.TestCase):
    def test_chunk_text_overlaps(self) -> None:
        self.assertEqual(chunk_text("abcdefgh", chunk_size=5, overlap=2), ("abcde", "defgh", "gh"))

    def test_contexts_accept_dict_of_lists(self) -> None:
        record = {"entity_pages": {"wiki_context": ["wiki"]}, "search_results": {"search_context": ["search"]}}
        self.assertEqual(tuple(record_contexts(record)), ("wiki", "search"))


if __name__ == "__main__":
    unittest.main()
