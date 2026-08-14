"""Logical database partition placement for Milvus Lite.

Milvus Lite does not implement Milvus partitions.  The reproduction therefore
stores each logical partition in its own collection and uses this module to
keep only a configurable subset resident between retrieval batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def partition_collection_name(prefix: str, partition_id: int) -> str:
    if not prefix:
        raise ValueError("collection prefix must not be empty")
    if partition_id < 0:
        raise ValueError("partition_id must be non-negative")
    return f"{prefix}_p{partition_id:02d}"


@dataclass(frozen=True)
class PartitionResidencySnapshot:
    resident_partition_ids: tuple[int, ...]
    loads: int
    releases: int
    searches: int


class PartitionResidency:
    """Track resident logical partitions and choose the next hot set.

    Partitions producing the strongest search hits remain resident.  Existing
    residents win ties, avoiding needless transfers when scores are equal.
    """

    def __init__(self, partition_count: int, resident_count: int) -> None:
        if partition_count < 1:
            raise ValueError("partition_count must be positive")
        if not 1 <= resident_count <= partition_count:
            raise ValueError("resident_count must be between 1 and partition_count")
        self.partition_count = partition_count
        self.resident_count = resident_count
        self._resident = tuple(range(resident_count))
        self.loads = 0
        self.releases = 0
        self.searches = 0

    @property
    def resident_partition_ids(self) -> tuple[int, ...]:
        return self._resident

    def is_resident(self, partition_id: int) -> bool:
        return partition_id in self._resident

    def select_next(self, scores: Iterable[tuple[int, float]]) -> tuple[int, ...]:
        current = set(self._resident)
        ranked = sorted(
            scores,
            key=lambda item: (item[1], item[0] in current, -item[0]),
            reverse=True,
        )
        if len(ranked) != self.partition_count:
            raise ValueError("scores must contain exactly one entry per partition")
        ids = [partition_id for partition_id, _ in ranked]
        if len(set(ids)) != self.partition_count or set(ids) != set(range(self.partition_count)):
            raise ValueError("scores contain invalid or duplicate partition ids")
        return tuple(sorted(ids[: self.resident_count]))

    def apply(self, next_resident: Iterable[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        target = tuple(sorted(next_resident))
        if len(target) != self.resident_count or len(set(target)) != len(target):
            raise ValueError("next resident set has the wrong size")
        if any(item < 0 or item >= self.partition_count for item in target):
            raise ValueError("next resident set contains an invalid partition id")
        current = set(self._resident)
        desired = set(target)
        to_load = tuple(sorted(desired - current))
        to_release = tuple(sorted(current - desired))
        self.loads += len(to_load)
        self.releases += len(to_release)
        self._resident = target
        return to_load, to_release

    def record_cold_load(self) -> None:
        self.loads += 1

    def record_cold_release(self) -> None:
        self.releases += 1

    def record_search(self) -> None:
        self.searches += 1

    def snapshot(self) -> PartitionResidencySnapshot:
        return PartitionResidencySnapshot(
            resident_partition_ids=self._resident,
            loads=self.loads,
            releases=self.releases,
            searches=self.searches,
        )
