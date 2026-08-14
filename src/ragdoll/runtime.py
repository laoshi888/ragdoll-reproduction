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

    @property
    def waiting_seconds(self) -> float:
        """Queueing before retrieval plus queueing between both stages."""
        return (self.retrieval_started_at - self.arrived_at) + (
            self.generation_started_at - self.retrieval_finished_at
        )

    @property
    def retrieval_seconds(self) -> float:
        return self.retrieval_finished_at - self.retrieval_started_at

    @property
    def generation_seconds(self) -> float:
        return self.completed_at - self.generation_started_at


@dataclass(frozen=True)
class RuntimeResult:
    responses: tuple[GeneratedResponse, ...]
    timings: tuple[RuntimeTiming, ...]
    retrieval_batch_sizes: tuple[int, ...]
    generation_batch_sizes: tuple[int, ...]


def _take_batch(queue: Queue, first: object, selector: BatchSelector, stage: str) -> tuple[list, bool]:
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


def _enqueue_at_arrival_times(
    requests: Sequence[RAGRequest], queue: Queue, arrivals: dict[int, float], arrival_scale: float
) -> tuple[RAGRequest, ...]:
    if arrival_scale < 0:
        raise ValueError("arrival_scale must be non-negative")
    ordered = tuple(sorted(requests, key=lambda request: request.arrival_time))
    if not ordered:
        queue.put(None)
        return ordered
    wall_start = time.monotonic()
    first_arrival = ordered[0].arrival_time
    for request in ordered:
        delay = (request.arrival_time - first_arrival) * arrival_scale - (time.monotonic() - wall_start)
        if delay > 0:
            time.sleep(delay)
        arrivals[request.request_id] = time.monotonic()
        queue.put(request)
    queue.put(None)
    return ordered


class SerialRAGRunner:
    """Run retrieval followed by generation on one shared batch at a time."""

    def __init__(self, *, retriever: Retriever, generator: Generator, selector: BatchSelector) -> None:
        self._retriever = retriever
        self._generator = generator
        self._selector = selector

    def run(self, requests: Sequence[RAGRequest], *, arrival_scale: float = 1.0) -> RuntimeResult:
        queue: Queue[RAGRequest | None] = Queue()
        arrivals: dict[int, float] = {}
        responses: dict[int, GeneratedResponse] = {}
        timings: dict[int, RuntimeTiming] = {}
        batches: list[int] = []

        def worker() -> None:
            while True:
                first = queue.get()
                if first is None:
                    return
                batch, stop_seen = _take_batch(queue, first, self._selector, "serial")
                retrieval_started = time.monotonic()
                retrieved = tuple(self._retriever.retrieve(batch))
                retrieval_finished = time.monotonic()
                generation_started = time.monotonic()
                generated = tuple(self._generator.generate(retrieved))
                completed = time.monotonic()
                if {item.request.request_id for item in retrieved} != {item.request_id for item in batch}:
                    raise RuntimeError("retriever must return exactly one item per request")
                if {item.request_id for item in generated} != {item.request_id for item in batch}:
                    raise RuntimeError("generator must return exactly one response per request")
                by_id = {item.request_id: item for item in generated}
                batches.append(len(batch))
                for request in batch:
                    responses[request.request_id] = by_id[request.request_id]
                    timings[request.request_id] = RuntimeTiming(
                        request_id=request.request_id,
                        arrived_at=arrivals[request.request_id],
                        retrieval_started_at=retrieval_started,
                        retrieval_finished_at=retrieval_finished,
                        generation_started_at=generation_started,
                        completed_at=completed,
                    )
                if stop_seen:
                    return

        thread = Thread(target=worker, name="ragdoll-serial", daemon=True)
        thread.start()
        ordered = _enqueue_at_arrival_times(requests, queue, arrivals, arrival_scale)
        thread.join()
        if not ordered:
            return RuntimeResult((), (), (), ())
        expected_ids = {request.request_id for request in ordered}
        if set(responses) != expected_ids or set(timings) != expected_ids:
            raise RuntimeError("serial pipeline did not complete every request")
        return RuntimeResult(
            responses=tuple(responses[request.request_id] for request in ordered),
            timings=tuple(timings[request.request_id] for request in ordered),
            retrieval_batch_sizes=tuple(batches),
            generation_batch_sizes=tuple(batches),
        )


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
        retrieve_queue: Queue[RAGRequest | None] = Queue()
        generate_queue: Queue[tuple[RetrievedRequest, float, float] | None] = Queue()
        timings: dict[int, RuntimeTiming] = {}
        responses: dict[int, GeneratedResponse] = {}
        retrieval_batches: list[int] = []
        generation_batches: list[int] = []
        arrivals: dict[int, float] = {}
        lock = Lock()

        def retrieval_worker() -> None:
            while True:
                first = retrieve_queue.get()
                if first is None:
                    generate_queue.put(None)
                    return
                batch, stop_seen = _take_batch(retrieve_queue, first, self._retrieval_selector, "retrieval")
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
                batch, stop_seen = _take_batch(generate_queue, first, self._generation_selector, "generation")
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
        ordered = _enqueue_at_arrival_times(requests, retrieve_queue, arrivals, arrival_scale)
        retrieval_thread.join()
        generation_thread.join()
        if not ordered:
            return RuntimeResult((), (), (), ())
        expected_ids = {request.request_id for request in ordered}
        if set(responses) != expected_ids or set(timings) != expected_ids:
            raise RuntimeError("pipeline did not complete every request")
        return RuntimeResult(
            responses=tuple(responses[request.request_id] for request in ordered),
            timings=tuple(timings[request.request_id] for request in ordered),
            retrieval_batch_sizes=tuple(retrieval_batches),
            generation_batch_sizes=tuple(generation_batches),
        )
