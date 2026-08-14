"""Run the profiled two-worker RAG pipeline on AutoDL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragdoll.backends.milvus import MilvusRetriever  # noqa: E402
from ragdoll.backends.vllm import VLLMGenerator  # noqa: E402
from ragdoll.batching import ProfiledBatchSelector, ProfiledSerialBatchSelector  # noqa: E402
from ragdoll.contracts import ProfileStore, RAGRequest  # noqa: E402
from ragdoll.placement import load_placement_profiles, select_fastest_feasible  # noqa: E402
from ragdoll.runtime import PipelinedRAGRunner, SerialRAGRunner  # noqa: E402
from ragdoll.simulator import generate_poisson_workload  # noqa: E402


def _fixed_batch_selector(batch_size: int) -> Callable[[str, int], int]:
    if batch_size < 1:
        raise ValueError("static_batch_size must be positive")
    return lambda _stage, backlog: min(batch_size, backlog)


def _build_generator(cfg: dict):
    backend = cfg["run"].get("generator_backend", "vllm")
    if backend == "vllm":
        return VLLMGenerator(
            model=cfg["models"]["generator"],
            max_new_tokens=cfg["run"]["max_new_tokens"],
            **cfg["vllm"],
        )
    if backend == "flexllmgen":
        from ragdoll.backends.flexllmgen import FlexLLMGenerator

        flex = cfg["flexllmgen"]
        profile_path = Path(flex["placement_profile"])
        if not profile_path.is_absolute():
            profile_path = PROJECT_ROOT / profile_path
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


def _summary(policy: str, result, elapsed_wall_seconds: float) -> dict[str, object]:
    latencies = sorted(item.latency_seconds for item in result.timings)
    waiting = [item.waiting_seconds for item in result.timings]
    retrieval = [item.retrieval_seconds for item in result.timings]
    generation = [item.generation_seconds for item in result.timings]
    return {
        "policy": policy,
        "request_count": len(result.responses),
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "p95_latency_seconds": latencies[round(0.95 * (len(latencies) - 1))],
        "mean_waiting_seconds": sum(waiting) / len(waiting),
        "mean_retrieval_seconds": sum(retrieval) / len(retrieval),
        "mean_generation_seconds": sum(generation) / len(generation),
        "wall_seconds": elapsed_wall_seconds,
        "retrieval_batches": result.retrieval_batch_sizes,
        "generation_batches": result.generation_batch_sizes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "real_small.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the configured result path without changing the experiment config.",
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=float,
        help="Override the FlexLLMGen placement budget for a paired comparison.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("serial", "static", "adaptive"),
        default=("adaptive",),
        help="Policies to run against the same deterministic arrival sequence.",
    )
    args = parser.parse_args()
    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.max_gpu_memory_gib is not None:
        if cfg["run"].get("generator_backend", "vllm") != "flexllmgen":
            raise ValueError("--max-gpu-memory-gib is only valid for the FlexLLMGen backend")
        cfg["flexllmgen"]["max_gpu_memory_gib"] = args.max_gpu_memory_gib
    questions = [json.loads(line)["question"] for line in Path(cfg["artifacts"]["workload_questions"]).read_text(encoding="utf-8").splitlines()]
    arrivals = generate_poisson_workload(seed=cfg["run"]["seed"], requests_per_phase=cfg["run"]["requests_per_phase"], arrival_rates_per_minute=cfg["run"]["arrival_rates_per_minute"])
    requests = tuple(RAGRequest(item.request_id, questions[item.request_id], item.arrival_time) for item in arrivals)
    profiles = ProfileStore.load(Path(cfg["artifacts"]["profiles"]))
    retriever = MilvusRetriever(uri=cfg["milvus"]["uri"], collection=cfg["milvus"]["collection"], embedder_name=cfg["models"]["embedder"], top_k=cfg["run"]["top_k"])
    generator = _build_generator(cfg)
    try:
        output: dict[str, object] = {"run": cfg["run"]["name"], "policies": {}}
        for policy in args.policies:
            if policy == "serial":
                selector = ProfiledSerialBatchSelector(
                    tuple(cfg["scheduler"]["generation_batch_candidates"]),
                    profiles,
                    cfg["scheduler"]["static_batch_size"],
                )
                runner = SerialRAGRunner(
                    retriever=retriever,
                    generator=generator,
                    selector=selector,
                )
            elif policy == "static":
                selector = _fixed_batch_selector(cfg["scheduler"]["static_batch_size"])
                runner = PipelinedRAGRunner(
                    retriever=retriever,
                    generator=generator,
                    retrieval_selector=selector,
                    generation_selector=selector,
                )
            else:
                retrieval_selector = ProfiledBatchSelector(
                    tuple(cfg["scheduler"]["retrieval_batch_candidates"]),
                    profiles,
                    cfg["scheduler"]["static_batch_size"],
                )
                generation_selector = ProfiledBatchSelector(
                    tuple(cfg["scheduler"]["generation_batch_candidates"]),
                    profiles,
                    cfg["scheduler"]["static_batch_size"],
                )
                runner = PipelinedRAGRunner(
                    retriever=retriever,
                    generator=generator,
                    retrieval_selector=retrieval_selector,
                    generation_selector=generation_selector,
                )
            started = time.monotonic()
            result = runner.run(requests)
            output["policies"][policy] = _summary(policy, result, time.monotonic() - started)
            print(f"completed policy={policy}")
        path = args.output or Path(cfg["artifacts"]["result"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
    finally:
        generator.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
