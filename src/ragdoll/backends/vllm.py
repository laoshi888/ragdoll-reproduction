"""vLLM generator adapter, intentionally imported only when constructed."""

from __future__ import annotations

from typing import Sequence

from ..contracts import GeneratedResponse, RetrievedRequest


def _prompt(item: RetrievedRequest) -> str:
    context = "\n\n".join(item.contexts)
    return (
        "Answer the question using only the supplied context. "
        "If the context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {item.request.question}\nAnswer:"
    )


class VLLMGenerator:
    """Batched, non-streaming vLLM adapter suitable for the generation worker."""

    def __init__(
        self,
        *,
        model: str,
        max_new_tokens: int,
        gpu_memory_utilization: float = 0.75,
        max_num_seqs: int = 16,
    ) -> None:
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise RuntimeError("VLLMGenerator requires vllm; run it on AutoDL.") from error
        self._llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )
        self._sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def generate(self, requests: Sequence[RetrievedRequest]) -> Sequence[GeneratedResponse]:
        outputs = self._llm.generate([_prompt(item) for item in requests], self._sampling_params)
        return tuple(
            GeneratedResponse(request_id=item.request.request_id, text=output.outputs[0].text)
            for item, output in zip(requests, outputs, strict=True)
        )
