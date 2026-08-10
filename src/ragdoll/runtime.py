"""Two-worker RAG execution pipeline with no ML-framework imports.

The retriever and generator are injected implementations.  This lets the same
queueing logic run against tiny test doubles locally and Milvus/vLLM on AutoDL.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread
import time
from typing import Callable, Sequence

from .contracts import GeneratedResponse, Generator, RAGRequest, RetrievedRequest, Retriever


BatchSelector = Callable[[str, int], int]


@dataclass(frozen=True)
class RuntimeTiming:
    request_id: int
    arrived_at: float
    retrieval_started_at: float
    retrieval_finished_at: float
    generation_started_at: float
    completed_at: float

    @property
    def latency_seconds(self) -> float:
        return self.completed_at - self.arrived_at


@dataclass(frozen=True)
class RuntimeResult:
    responses: tuple[GeneratedResponse, ...]
    timings: tuple[RuntimeTiming, ...]
    retrieval_batch_sizes: tuple[int, ...]
    generation_batch_sizes: tuple[int, ...]


class PipelinedRAGRunner:
    """Run independent retrieval and generation batches in two worker threads."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        generator: Generator,
        retrieval_selector: BatchSelector,
        generation_selector: BatchSelector,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._retrieval_selector = retrieval_selector
        self._generation_selector = generation_selector

    def run(self, requests: Sequence[RAGRequest], *, arrival_scale: float = 1.0) -> RuntimeResult:
        """Execute a workload with arrival timestamps relative to its first item.

        Set ``arrival_scale=0`` only for CPU unit tests.  Real experiments use
        one to preserve the configured Poisson arrival process.
        """
        if arrival_scale < 0:
            raise ValueError("arrival_scale must be non-negative")
        ordered = tuple(sorted(requests, key=lambda request: request.arrival_time))
        if not ordered:
            return RuntimeResult((), (), (), ())
        retrieve_queue: Queue[RAGRequest | None] = Queue()
        generate_queue: Queue[tuple[RetrievedRequest, float, float] | None] = Queue()
        timings: dict[int, RuntimeTiming] = {}
        responses: dict[int, GeneratedResponse] = {}
        retrieval_batches: list[int] = []
        generation_batches: list[int] = []
        arrivals: dict[int, float] = {}
        lock = Lock()

        def take_batch(queue: Queue, first: object, selector: BatchSelector, stage: str) -> tuple[list, bool]:
            pending = queue.qsize() + 1
            size = selector(stage, pending)
            if size < 1:
                raise ValueError(f"{stage} selector returned a non-positive batch size")
            batch = [first]
            stop_seen = False
            while len(batch) < size:
                try:
                    item = queue.get_nowait()
                except Empty:
                    break
                if item is None:
                    stop_seen = True
                    break
                batch.append(item)
            return batch, stop_seen

        def retrieval_worker() -> None:
            while True:
                first = retrieve_queue.get()
                if first is None:
                    generate_queue.put(None)
                    return
                batch, stop_seen = take_batch(retrieve_queue, first, self._retrieval_selector, "retrieval")
                started = time.monotonic()
                retrieved = tuple(self._retriever.retrieve(batch))
                finished = time.monotonic()
                if {item.request.request_id for item in retrieved} != {item.request_id for item in batch}:
                    raise RuntimeError("retriever must return exactly one item per request")
                with lock:
                    retrieval_batches.append(len(batch))
                for item in retrieved:
                    generate_queue.put((item, started, finished))
                if stop_seen:
                    generate_queue.put(None)
                    return

        def generation_worker() -> None:
            while True:
                first = generate_queue.get()
                if first is None:
                    return
                batch, stop_seen = take_batch(generate_queue, first, self._generation_selector, "generation")
                retrieved = [item[0] for item in batch]
                started = time.monotonic()
                generated = tuple(self._generator.generate(retrieved))
                finished = time.monotonic()
                if {item.request_id for item in generated} != {item.request.request_id for item in retrieved}:
                    raise RuntimeError("generator must return exactly one response per retrieved request")
                by_id = {item.request_id: item for item in generated}
                with lock:
                    generation_batches.append(len(batch))
                    for item, retrieval_started, retrieval_finished in batch:
                        request_id = item.request.request_id
                        responses[request_id] = by_id[request_id]
                        timings[request_id] = RuntimeTiming(
                            request_id=request_id,
                            arrived_at=arrivals[request_id],
                            retrieval_started_at=retrieval_started,
                            retrieval_finished_at=retrieval_finished,
                            generation_started_at=started,
                            completed_at=finished,
                        )
                if stop_seen:
                    return

        retrieval_thread = Thread(target=retrieval_worker, name="ragdoll-retrieval", daemon=True)
        generation_thread = Thread(target=generation_worker, name="ragdoll-generation", daemon=True)
        retrieval_thread.start()
        generation_thread.start()
        wall_start = time.monotonic()
        first_arrival = ordered[0].arrival_time
        for request in ordered:
            delay = (request.arrival_time - first_arrival) * arrival_scale - (time.monotonic() - wall_start)
            if delay > 0:
                time.sleep(delay)
            arrivals[request.request_id] = time.monotonic()
            retrieve_queue.put(request)
        retrieve_queue.put(None)
        retrieval_thread.join()
        generation_thread.join()
        expected_ids = {request.request_id for request in ordered}
        if set(responses) != expected_ids or set(timings) != expected_ids:
            raise RuntimeError("pipeline did not complete every request")
        return RuntimeResult(
            responses=tuple(responses[request.request_id] for request in ordered),
            timings=tuple(timings[request.request_id] for request in ordered),
            retrieval_batch_sizes=tuple(retrieval_batches),
            generation_batch_sizes=tuple(generation_batches),
        )
