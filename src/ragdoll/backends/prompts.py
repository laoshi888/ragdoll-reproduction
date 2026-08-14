"""Shared prompt construction for generation backends."""

from __future__ import annotations

from ..contracts import RetrievedRequest


def build_rag_prompt(item: RetrievedRequest) -> str:
    context = "\n\n".join(item.contexts)
    return (
        "Answer the question using only the supplied context. "
        "If the context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {item.request.question}\nAnswer:"
    )
