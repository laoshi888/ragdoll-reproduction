from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdoll.contracts import GeneratedResponse, RAGRequest, RetrievedRequest  # noqa: E402
from ragdoll.runtime import PipelinedRAGRunner, RuntimeTiming, SerialRAGRunner  # noqa: E402


class EchoRetriever:
    def retrieve(self, requests):
        return [RetrievedRequest(request, (f"context:{request.question}",)) for request in requests]


class EchoGenerator:
    def generate(self, requests):
        return [GeneratedResponse(item.request.request_id, item.contexts[0]) for item in requests]


class RuntimeTests(unittest.TestCase):
    def test_timing_breakdown_sums_to_end_to_end_latency(self) -> None:
        timing = RuntimeTiming(0, 1.0, 2.0, 4.0, 5.0, 8.0)
        self.assertEqual(timing.waiting_seconds, 2.0)
        self.assertEqual(timing.retrieval_seconds, 2.0)
        self.assertEqual(timing.generation_seconds, 3.0)
        self.assertEqual(
            timing.waiting_seconds + timing.retrieval_seconds + timing.generation_seconds,
            timing.latency_seconds,
        )

    def test_serial_pipeline_uses_the_same_batch_for_both_stages(self) -> None:
        requests = tuple(RAGRequest(index, f"q{index}", 0.0) for index in range(6))
        runner = SerialRAGRunner(
            retriever=EchoRetriever(),
            generator=EchoGenerator(),
            selector=lambda _stage, backlog: min(3, backlog),
        )
        result = runner.run(requests, arrival_scale=0)
        self.assertEqual([item.request_id for item in result.responses], list(range(6)))
        self.assertEqual(result.retrieval_batch_sizes, result.generation_batch_sizes)

    def test_two_worker_pipeline_preserves_request_identity(self) -> None:
        requests = tuple(RAGRequest(index, f"q{index}", float(index)) for index in range(8))
        runner = PipelinedRAGRunner(
            retriever=EchoRetriever(),
            generator=EchoGenerator(),
            retrieval_selector=lambda _stage, backlog: min(2, backlog),
            generation_selector=lambda _stage, backlog: min(3, backlog),
        )
        result = runner.run(requests, arrival_scale=0)
        self.assertEqual([item.request_id for item in result.responses], list(range(8)))
        self.assertTrue(all(item.latency_seconds >= 0 for item in result.timings))
        self.assertTrue(result.retrieval_batch_sizes)
        self.assertTrue(result.generation_batch_sizes)

    def test_empty_workload_does_not_start_work(self) -> None:
        runner = PipelinedRAGRunner(
            retriever=EchoRetriever(),
            generator=EchoGenerator(),
            retrieval_selector=lambda _stage, backlog: backlog,
            generation_selector=lambda _stage, backlog: backlog,
        )
        self.assertEqual(runner.run((), arrival_scale=0).responses, ())


if __name__ == "__main__":
    unittest.main()
