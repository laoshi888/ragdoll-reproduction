"""Backend-neutral data contracts for the real RAGDoll reproduction.

Imports in this module are intentionally limited to the standard library.  A
local developer machine can import and test scheduling code without loading
PyTorch, a model checkpoint, Milvus, or CUDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class RAGRequest:
    request_id: int
    question: str
    arrival_time: float


@dataclass(frozen=True)
class RetrievedRequest:
    request: RAGRequest
    contexts: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedResponse:
    request_id: int
    text: str


class Retriever(Protocol):
    """CPU/database stage. Implementations may use Milvus or a test double."""

    def retrieve(self, requests: Sequence[RAGRequest]) -> Sequence[RetrievedRequest]:
        """Return one retrieved item for every input request, preserving IDs."""


class Generator(Protocol):
    """GPU stage. Implementations may use vLLM or an offloading backend."""

    def generate(self, requests: Sequence[RetrievedRequest]) -> Sequence[GeneratedResponse]:
        """Return one generated response for every retrieved request, preserving IDs."""


@dataclass(frozen=True)
class ProfileSample:
    """One timing observation used by the online batch selector."""

    stage: str
    batch_size: int
    elapsed_seconds: float


class ProfileStore:
    """Small JSON persistence for profiling data; no database dependency."""

    def __init__(self, samples: Sequence[ProfileSample] = ()) -> None:
        self._samples = list(samples)

    @property
    def samples(self) -> tuple[ProfileSample, ...]:
        return tuple(self._samples)

    def add(self, sample: ProfileSample) -> None:
        if sample.batch_size < 1 or sample.elapsed_seconds < 0:
            raise ValueError("batch size must be positive and elapsed time non-negative")
        self._samples.append(sample)

    def mean_seconds(self, stage: str, batch_size: int) -> float:
        values = [
            item.elapsed_seconds
            for item in self._samples
            if item.stage == stage and item.batch_size == batch_size
        ]
        if not values:
            raise LookupError(f"no profile samples for {stage=}, {batch_size=}")
        return sum(values) / len(values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(item) for item in self._samples], indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ProfileStore":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(ProfileSample(**item) for item in raw)
