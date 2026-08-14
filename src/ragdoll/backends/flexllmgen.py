"""FlexLLMGen adapter for the real RAG pipeline, imported only on AutoDL."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..contracts import GeneratedResponse, RetrievedRequest
from .prompts import build_rag_prompt


class FlexLLMGenerator:
    """Run fixed-capacity OPT batches with profiled GPU/CPU memory placement.

    FlexLLMGen fixes its effective batch capacity when the model is built.  A
    smaller online batch is padded by repeating its last prompt, and padding
    outputs are discarded.  Requests larger than the configured capacity are
    rejected so scheduling mistakes remain visible.
    """

    def __init__(
        self,
        *,
        model: str,
        max_new_tokens: int,
        prompt_length: int,
        percent: tuple[int, int, int, int, int, int],
        weights_path: str,
        offload_dir: str,
        gpu_batch_size: int = 1,
        num_gpu_batches: int = 1,
        overlap: bool = True,
        pin_weight: bool = True,
        warmup: bool = True,
    ) -> None:
        if max_new_tokens < 1 or prompt_length < 1:
            raise ValueError("token lengths must be positive")
        if gpu_batch_size < 1 or num_gpu_batches < 1:
            raise ValueError("batch dimensions must be positive")
        if len(percent) != 6 or any(value < 0 or value > 100 for value in percent):
            raise ValueError("percent must contain six values in [0, 100]")
        for gpu, cpu in zip(percent[::2], percent[1::2], strict=True):
            if gpu + cpu > 100:
                raise ValueError("each GPU/CPU percentage pair must sum to at most 100")
        try:
            from flexllmgen.compression import CompressionConfig
            from flexllmgen.flex_opt import OptLM, Policy
            from flexllmgen.opt_config import get_opt_config
            from flexllmgen.pytorch_backend import (
                TorchDevice,
                TorchDisk,
                TorchMixedDevice,
            )
            from flexllmgen.utils import ExecutionEnv
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "FlexLLMGenerator requires the AutoDL FlexLLMGen environment."
            ) from error

        Path(offload_dir).mkdir(parents=True, exist_ok=True)
        gpu = TorchDevice("cuda:0")
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir)
        self._env = ExecutionEnv(
            gpu=gpu,
            cpu=cpu,
            disk=disk,
            mixed=TorchMixedDevice([gpu, cpu, disk]),
        )
        compression_weight = CompressionConfig(
            num_bits=4, group_size=64, group_dim=0, symmetric=False
        )
        compression_cache = CompressionConfig(
            num_bits=4, group_size=64, group_dim=2, symmetric=False
        )
        policy = Policy(
            gpu_batch_size,
            num_gpu_batches,
            *percent,
            overlap,
            True,
            pin_weight,
            False,
            1.0,
            False,
            compression_weight,
            False,
            compression_cache,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        self._model = OptLM(get_opt_config(model), self._env, weights_path, policy)
        self._prompt_length = prompt_length
        self._max_new_tokens = max_new_tokens
        self._capacity = gpu_batch_size * num_gpu_batches
        if warmup:
            warmup_ids = self._tokenize_to_capacity(("Warmup",))
            self._model.generate(warmup_ids, max_new_tokens=1, verbose=0)

    def _tokenize_to_capacity(self, prompts: Sequence[str]) -> list[list[int]]:
        encoded = self._tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self._prompt_length,
        )
        input_ids = [list(row) for row in encoded["input_ids"]]
        while len(input_ids) < self._capacity:
            input_ids.append(list(input_ids[-1]))
        return input_ids

    def generate(self, requests: Sequence[RetrievedRequest]) -> Sequence[GeneratedResponse]:
        if not requests:
            return ()
        if len(requests) > self._capacity:
            raise ValueError(
                f"FlexLLMGen batch size {len(requests)} exceeds fixed capacity {self._capacity}"
            )
        input_ids = self._tokenize_to_capacity(
            tuple(build_rag_prompt(item) for item in requests)
        )
        output_ids = self._model.generate(
            input_ids,
            max_new_tokens=self._max_new_tokens,
            stop=self._tokenizer.eos_token_id,
            verbose=0,
        )
        generated_ids = [row[self._prompt_length :] for row in output_ids]
        texts = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return tuple(
            GeneratedResponse(request_id=item.request.request_id, text=texts[index])
            for index, item in enumerate(requests)
        )

    def close(self) -> None:
        """Release model tensors before stopping FlexLLMGen copy workers."""
        if self._model is not None:
            self._model = None
        if self._env is not None:
            self._env.close_copy_threads()
            self._env = None
