"""Load the measured logical-partition residency profile."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PartitionResidencyProfile:
    resident_partitions: int
    mean_latency_seconds: float
    mean_retrieval_seconds: float
    mean_loads: float
    mean_releases: float


def load_partition_residency_profiles(path: Path) -> tuple[PartitionResidencyProfile, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("summary")
    if not isinstance(rows, list) or not rows:
        raise ValueError("partition residency profile needs a non-empty summary list")
    profiles: list[PartitionResidencyProfile] = []
    for row in rows:
        profile = PartitionResidencyProfile(
            resident_partitions=int(row["resident_partitions"]),
            mean_latency_seconds=float(row["mean_latency_seconds"]),
            mean_retrieval_seconds=float(row["mean_retrieval_seconds"]),
            mean_loads=float(row["mean_loads"]),
            mean_releases=float(row["mean_releases"]),
        )
        if profile.resident_partitions < 1 or profile.mean_latency_seconds <= 0:
            raise ValueError("partition residency profile contains an invalid value")
        profiles.append(profile)
    if len({profile.resident_partitions for profile in profiles}) != len(profiles):
        raise ValueError("partition residency profile has duplicate residency counts")
    return tuple(profiles)


def select_fastest_residency(
    profiles: tuple[PartitionResidencyProfile, ...], partition_count: int
) -> PartitionResidencyProfile:
    feasible = [profile for profile in profiles if profile.resident_partitions <= partition_count]
    if not feasible:
        raise ValueError("partition residency profile has no feasible candidate")
    return min(feasible, key=lambda profile: profile.mean_latency_seconds)
