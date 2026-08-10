"""Dependency-free helpers for constructing a small TriviaQA RAG corpus."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> tuple[str, ...]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than it")
    cleaned = " ".join(text.split())
    if not cleaned:
        return ()
    step = chunk_size - overlap
    return tuple(cleaned[start : start + chunk_size] for start in range(0, len(cleaned), step))


def record_contexts(record: Mapping[str, object]) -> Iterable[str]:
    """Yield wiki/search evidence despite HF Sequence's two possible layouts."""
    for name, field in (("entity_pages", "wiki_context"), ("search_results", "search_context")):
        source = record.get(name, ())
        if isinstance(source, Mapping):
            values = source.get(field, ())
            yield from (value for value in values if isinstance(value, str) and value.strip())
        elif isinstance(source, list):
            for item in source:
                if isinstance(item, Mapping):
                    value = item.get(field)
                    if isinstance(value, str) and value.strip():
                        yield value
