"""Offline-profiled GPU/CPU placement selection for the generation backend."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PlacementProfile:
    """Measured cost of one valid FlexLLMGen memory placement."""

    name: str
    percent: tuple[int, int, int, int, int, int]
    peak_gpu_memory_gib: float
    total_latency_seconds: float
    decode_throughput_tokens_per_second: float


def load_placement_profiles(path: Path) -> tuple[PlacementProfile, ...]:
    """Load measured placements from the versioned JSON profile."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    profiles: list[PlacementProfile] = []
    for item in raw["placements"]:
        percent = tuple(item["percent"])
        if len(percent) != 6:
            raise ValueError(f"placement {item['name']!r} must contain six percentages")
        profiles.append(
            PlacementProfile(
                name=item["name"],
                percent=percent,
                peak_gpu_memory_gib=item["peak_gpu_memory_gib"],
                total_latency_seconds=item["total_latency_seconds"],
                decode_throughput_tokens_per_second=item[
                    "decode_throughput_tokens_per_second"
                ],
            )
        )
    return tuple(profiles)


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
