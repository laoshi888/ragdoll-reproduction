from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.backends.flexllmgen import FlexLLMGenerator  # noqa: E402
from ragdoll.contracts import RAGRequest, RetrievedRequest  # noqa: E402


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompts, *, padding, truncation, max_length):
        self.prompts = list(prompts)
        return {"input_ids": [[1] * max_length for _ in prompts]}

    def batch_decode(self, rows, *, skip_special_tokens):
        return [" ".join(str(value) for value in row) for row in rows]


class FakeModel:
    def __init__(self) -> None:
        self.batch_size = 0

    def generate(self, input_ids, *, max_new_tokens, stop, verbose):
        self.batch_size = len(input_ids)
        return [row + [10 + index] * max_new_tokens for index, row in enumerate(input_ids)]


def make_generator(capacity: int = 2) -> FlexLLMGenerator:
    generator = object.__new__(FlexLLMGenerator)
    generator._tokenizer = FakeTokenizer()
    generator._model = FakeModel()
    generator._prompt_length = 4
    generator._max_new_tokens = 2
    generator._capacity = capacity
    return generator


class FlexLLMGeneratorTests(unittest.TestCase):
    def test_pads_to_fixed_capacity_and_discards_padding_output(self) -> None:
        generator = make_generator()
        request = RAGRequest(7, "question", 0.0)
        responses = generator.generate((RetrievedRequest(request, ("context",)),))

        self.assertEqual(generator._model.batch_size, 2)
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].request_id, 7)
        self.assertEqual(responses[0].text, "10 10")
        self.assertIn("context", generator._tokenizer.prompts[0])
        self.assertIn("question", generator._tokenizer.prompts[0])

    def test_rejects_batch_larger_than_fixed_capacity(self) -> None:
        generator = make_generator(capacity=1)
        items = tuple(
            RetrievedRequest(RAGRequest(index, "q", 0.0), ("c",)) for index in range(2)
        )
        with self.assertRaises(ValueError):
            generator.generate(items)


if __name__ == "__main__":
    unittest.main()
