"""Offline-profiled GPU/CPU placement selection for the generation backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementProfile:
    """Measured cost of one valid FlexLLMGen memory placement."""

    name: str
    percent: tuple[int, int, int, int, int, int]
    peak_gpu_memory_gib: float
    total_latency_seconds: float
    decode_throughput_tokens_per_second: float


def select_fastest_feasible(
    profiles: tuple[PlacementProfile, ...],
    max_gpu_memory_gib: float,
) -> PlacementProfile:
    """Return the lowest-latency measured placement fitting a GPU-memory budget.

    This deliberately chooses only from measured profiles.  RAGDoll's offline
    active profiling avoids extrapolating a memory layout from model size alone.
    """

    if max_gpu_memory_gib <= 0:
        raise ValueError("max_gpu_memory_gib must be positive")
    feasible = tuple(profile for profile in profiles if profile.peak_gpu_memory_gib <= max_gpu_memory_gib)
    if not feasible:
        raise ValueError(f"no profiled placement fits {max_gpu_memory_gib:.3f} GiB")
    return min(feasible, key=lambda profile: (profile.total_latency_seconds, profile.name))
