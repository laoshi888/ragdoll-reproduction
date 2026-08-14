"""Offline-profiled execution-topology selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class TopologyProfile:
    placement: str
    topology: str
    mean_latency_seconds: float
    p95_latency_seconds: float


def load_topology_profiles(path: Path) -> tuple[TopologyProfile, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(TopologyProfile(**item) for item in raw["profiles"])


def select_fastest_topology(
    profiles: tuple[TopologyProfile, ...], placement: str
) -> TopologyProfile:
    """Choose the lowest measured mean-latency topology for one placement."""

    candidates = tuple(item for item in profiles if item.placement == placement)
    if not candidates:
        raise ValueError(f"no topology profile exists for placement {placement!r}")
    invalid = tuple(item.topology for item in candidates if item.topology not in {"serial", "static", "adaptive"})
    if invalid:
        raise ValueError(f"unsupported profiled topology: {invalid[0]}")
    return min(candidates, key=lambda item: (item.mean_latency_seconds, item.topology))
