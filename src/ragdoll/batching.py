"""Profile-driven batch selectors shared by simulated and real pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import ProfileStore


@dataclass(frozen=True)
class ProfiledBatchSelector:
    """Pick a feasible batch from measured stage timings.

    The scoring function is the same mean-completion-time approximation used
    in the simulator.  Before a stage has enough offline profile data, a small
    conservative fallback batch is used instead of guessing from hardware.
    """

    candidates: tuple[int, ...]
    profiles: ProfileStore
    fallback_batch_size: int = 1

    def __call__(self, stage: str, backlog: int) -> int:
        if backlog < 1:
            raise ValueError("backlog must be positive")
        feasible = tuple(size for size in self.candidates if size <= backlog)
        if not feasible:
            return 1
        timed: list[tuple[int, float]] = []
        for size in feasible:
            try:
                timed.append((size, self.profiles.mean_seconds(stage, size)))
            except LookupError:
                continue
        if not timed:
            return min(backlog, self.fallback_batch_size)
        return min(
            timed,
            key=lambda item: (item[1] * (math.ceil(backlog / item[0]) + 1) / 2, item[0]),
        )[0]
