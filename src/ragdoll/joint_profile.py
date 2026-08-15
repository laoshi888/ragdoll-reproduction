"""Measured end-to-end RAG configuration selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class JointConfigurationProfile:
    name: str
    max_gpu_memory_gib: float
    placement: str
    resident_partitions: int
    topology: str
    median_latency_seconds: float
    mean_latency_seconds: float


def load_joint_configuration_profiles(path: Path) -> tuple[JointConfigurationProfile, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("summary")
    if not isinstance(rows, list) or not rows:
        raise ValueError("joint configuration profile needs a non-empty summary list")
    profiles: list[JointConfigurationProfile] = []
    for row in rows:
        profile = JointConfigurationProfile(
            name=str(row["name"]),
            max_gpu_memory_gib=float(row["max_gpu_memory_gib"]),
            placement=str(row["placement"]),
            resident_partitions=int(row["resident_partitions"]),
            topology=str(row["topology"]),
            median_latency_seconds=float(row["median_latency_seconds"]),
            mean_latency_seconds=float(row["mean_latency_seconds"]),
        )
        if (
            profile.max_gpu_memory_gib <= 0
            or profile.resident_partitions < 1
            or profile.median_latency_seconds <= 0
            or profile.topology not in {"serial", "static", "adaptive"}
        ):
            raise ValueError("joint configuration profile contains an invalid value")
        profiles.append(profile)
    if len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("joint configuration profile has duplicate names")
    return tuple(profiles)


def select_fastest_joint_configuration(
    profiles: tuple[JointConfigurationProfile, ...], max_gpu_memory_gib: float
) -> JointConfigurationProfile:
    if max_gpu_memory_gib <= 0:
        raise ValueError("max_gpu_memory_gib must be positive")
    feasible = [
        profile
        for profile in profiles
        if profile.max_gpu_memory_gib <= max_gpu_memory_gib
    ]
    if not feasible:
        raise ValueError(
            f"no measured joint configuration fits {max_gpu_memory_gib:.3f} GiB"
        )
    return min(
        feasible,
        key=lambda profile: (profile.median_latency_seconds, profile.mean_latency_seconds, profile.name),
    )
