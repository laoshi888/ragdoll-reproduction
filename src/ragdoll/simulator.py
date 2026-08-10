"""Deterministic discrete-event model of RAGDoll's scheduling design.

This module deliberately models timing rather than performing retrieval or LLM
inference.  It makes the paper's two independently batched pipelines testable
on a low-memory development machine before real backends are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import heapq
import math
import random
from typing import Iterable


class Policy(str, Enum):
    SERIAL = "serial"
    PIPELINED_STATIC = "pipelined_static"
    PIPELINED_ADAPTIVE = "pipelined_adaptive"


@dataclass(frozen=True)
class StageConfig:
    batch_candidates: tuple[int, ...]
    static_batch_size: int
    seconds_scale: float
    seconds_exponent: float
    partition_load_seconds: float = 0.0

    def duration(self, batch_size: int) -> float:
        """Measured-model surrogate T(B)=a*B^c plus fixed partition load."""
        return self.partition_load_seconds + self.seconds_scale * batch_size**self.seconds_exponent


@dataclass(frozen=True)
class SimulationConfig:
    retrieval: StageConfig
    generation: StageConfig


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_time: float


@dataclass(frozen=True)
class RequestTiming:
    request_id: int
    arrival_time: float
    retrieval_started_at: float
    retrieval_finished_at: float
    generation_started_at: float
    completed_at: float

    @property
    def waiting_seconds(self) -> float:
        return (self.retrieval_started_at - self.arrival_time) + (
            self.generation_started_at - self.retrieval_finished_at
        )

    @property
    def latency_seconds(self) -> float:
        return self.completed_at - self.arrival_time


@dataclass(frozen=True)
class SimulationResult:
    policy: Policy
    timings: tuple[RequestTiming, ...]
    retrieval_batches: tuple[int, ...]
    generation_batches: tuple[int, ...]

    @property
    def average_latency_seconds(self) -> float:
        return sum(item.latency_seconds for item in self.timings) / len(self.timings)

    @property
    def average_waiting_seconds(self) -> float:
        return sum(item.waiting_seconds for item in self.timings) / len(self.timings)


def generate_poisson_workload(
    *, requests_per_phase: int, arrival_rates_per_minute: Iterable[float], seed: int
) -> tuple[Request, ...]:
    """Create the paper-style piecewise Poisson request stream reproducibly."""
    rng = random.Random(seed)
    current_time = 0.0
    requests: list[Request] = []
    for rate_per_minute in arrival_rates_per_minute:
        if rate_per_minute <= 0:
            raise ValueError("arrival rates must be positive")
        for _ in range(requests_per_phase):
            current_time += rng.expovariate(rate_per_minute / 60.0)
            requests.append(Request(len(requests), current_time))
    return tuple(requests)


def choose_backlog_aware_batch(queue_size: int, stage: StageConfig) -> int:
    """Choose B minimizing the paper-inspired mean completion-time estimate.

    For each feasible B, approximate queued work as equally sized batches.  The
    resulting score is T(B) * (ceil(N/B) + 1) / 2; it captures the trade-off
    between larger-batch throughput and earlier completion for queued requests.
    """
    if queue_size < 1:
        raise ValueError("queue_size must be positive")
    candidates = [size for size in stage.batch_candidates if size <= queue_size]
    if not candidates:
        return 1
    return min(
        candidates,
        key=lambda size: (stage.duration(size) * (math.ceil(queue_size / size) + 1) / 2, size),
    )


def run_simulation(
    requests: Iterable[Request], config: SimulationConfig, policy: Policy
) -> SimulationResult:
    """Simulate serial or two-worker RAG service without allocating model data."""
    ordered_requests = tuple(sorted(requests, key=lambda item: item.arrival_time))
    if not ordered_requests:
        raise ValueError("at least one request is required")
    if policy is Policy.SERIAL:
        return _run_serial(ordered_requests, config)
    return _run_pipelined(ordered_requests, config, policy)


def _batch_size(queue_size: int, stage: StageConfig, policy: Policy) -> int:
    if policy is Policy.PIPELINED_ADAPTIVE:
        return choose_backlog_aware_batch(queue_size, stage)
    return min(queue_size, stage.static_batch_size)


def _run_serial(requests: tuple[Request, ...], config: SimulationConfig) -> SimulationResult:
    now = 0.0
    timings: list[RequestTiming] = []
    retrieval_batches: list[int] = []
    generation_batches: list[int] = []
    cursor = 0
    while cursor < len(requests):
        # A request cannot be processed before it arrives.  Form the serial
        # batch from the backlog that exists at this instant; if the service is
        # idle, advance to the next arrival rather than including future work.
        now = max(now, requests[cursor].arrival_time)
        available_end = cursor
        while available_end < len(requests) and requests[available_end].arrival_time <= now:
            available_end += 1
        batch_end = min(cursor + config.retrieval.static_batch_size, available_end)
        batch = requests[cursor:batch_end]
        retrieval_batches.append(len(batch))
        retrieval_started = now
        retrieval_finished = retrieval_started + config.retrieval.duration(len(batch))
        generation_started = retrieval_finished
        generation_finished = generation_started + config.generation.duration(len(batch))
        generation_batches.append(len(batch))
        timings.extend(
            RequestTiming(
                request_id=item.request_id,
                arrival_time=item.arrival_time,
                retrieval_started_at=retrieval_started,
                retrieval_finished_at=retrieval_finished,
                generation_started_at=generation_started,
                completed_at=generation_finished,
            )
            for item in batch
        )
        now = generation_finished
        cursor = batch_end
    return _result(Policy.SERIAL, timings, retrieval_batches, generation_batches)


def _run_pipelined(
    requests: tuple[Request, ...], config: SimulationConfig, policy: Policy
) -> SimulationResult:
    events: list[tuple[float, int, str, tuple[Request, ...]]] = []
    for request in requests:
        heapq.heappush(events, (request.arrival_time, request.request_id, "arrival", (request,)))
    retrieval_queue: list[Request] = []
    generation_queue: list[tuple[Request, float, float]] = []
    timings: dict[int, RequestTiming] = {}
    retrieval_batches: list[int] = []
    generation_batches: list[int] = []
    retrieval_busy = False
    generation_busy = False
    event_order = len(requests)

    def schedule_available_workers(now: float) -> None:
        nonlocal retrieval_busy, generation_busy, event_order
        if not retrieval_busy and retrieval_queue:
            size = _batch_size(len(retrieval_queue), config.retrieval, policy)
            batch = tuple(retrieval_queue[:size])
            del retrieval_queue[:size]
            retrieval_busy = True
            retrieval_batches.append(size)
            heapq.heappush(
                events,
                (now + config.retrieval.duration(size), event_order, "retrieval_done", batch),
            )
            event_order += 1
        if not generation_busy and generation_queue:
            size = _batch_size(len(generation_queue), config.generation, policy)
            batch = tuple(generation_queue[:size])
            del generation_queue[:size]
            generation_busy = True
            generation_batches.append(size)
            completion = now + config.generation.duration(size)
            heapq.heappush(events, (completion, event_order, "generation_done", batch))
            event_order += 1

    while events:
        now, _, event_type, batch = heapq.heappop(events)
        if event_type == "arrival":
            retrieval_queue.extend(batch)
        elif event_type == "retrieval_done":
            retrieval_busy = False
            for request in batch:
                retrieval_started = now - config.retrieval.duration(len(batch))
                generation_queue.append((request, retrieval_started, now))
        elif event_type == "generation_done":
            generation_busy = False
            generation_started = now - config.generation.duration(len(batch))
            for request, retrieval_started, retrieval_finished in batch:
                timings[request.request_id] = RequestTiming(
                    request_id=request.request_id,
                    arrival_time=request.arrival_time,
                    retrieval_started_at=retrieval_started,
                    retrieval_finished_at=retrieval_finished,
                    generation_started_at=generation_started,
                    completed_at=now,
                )
        schedule_available_workers(now)
    return _result(policy, list(timings.values()), retrieval_batches, generation_batches)


def _result(
    policy: Policy, timings: list[RequestTiming], retrieval_batches: list[int], generation_batches: list[int]
) -> SimulationResult:
    return SimulationResult(
        policy=policy,
        timings=tuple(sorted(timings, key=lambda item: item.request_id)),
        retrieval_batches=tuple(retrieval_batches),
        generation_batches=tuple(generation_batches),
    )
