"""Configuration-driven generator construction shared by real scripts."""

from __future__ import annotations

from pathlib import Path

from .backends.vllm import VLLMGenerator
from .placement import load_placement_profiles, select_fastest_feasible


def build_generator(cfg: dict, project_root: Path):
    """Construct the configured generation backend without changing runner code."""

    backend = cfg["run"].get("generator_backend", "vllm")
    if backend == "vllm":
        return VLLMGenerator(
            model=cfg["models"]["generator"],
            max_new_tokens=cfg["run"]["max_new_tokens"],
            **cfg["vllm"],
        )
    if backend == "flexllmgen":
        from .backends.flexllmgen import FlexLLMGenerator

        flex = cfg["flexllmgen"]
        profile_path = Path(flex["placement_profile"])
        if not profile_path.is_absolute():
            profile_path = project_root / profile_path
        selected = select_fastest_feasible(
            load_placement_profiles(profile_path),
            flex["max_gpu_memory_gib"],
        )
        print(
            f"selected placement={selected.name} percent={list(selected.percent)} "
            f"profiled_peak_gpu_gib={selected.peak_gpu_memory_gib:.4f}"
        )
        return FlexLLMGenerator(
            model=cfg["models"]["generator"],
            max_new_tokens=cfg["run"]["max_new_tokens"],
            prompt_length=flex["prompt_length"],
            percent=selected.percent,
            weights_path=flex["weights_path"],
            offload_dir=flex["offload_dir"],
            gpu_batch_size=flex["gpu_batch_size"],
            num_gpu_batches=flex["num_gpu_batches"],
            overlap=flex.get("overlap", True),
            pin_weight=flex.get("pin_weight", True),
            warmup=flex.get("warmup", True),
        )
    raise ValueError(f"unsupported generator backend: {backend}")
