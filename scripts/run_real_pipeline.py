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

from ragdoll.backend_factory import build_generator  # noqa: E402
from ragdoll.batching import ProfiledBatchSelector, ProfiledSerialBatchSelector  # noqa: E402
from ragdoll.contracts import ProfileStore, RAGRequest  # noqa: E402
from ragdoll.runtime import PipelinedRAGRunner, SerialRAGRunner  # noqa: E402
from ragdoll.retriever_factory import build_retriever  # noqa: E402
from ragdoll.partition_profile import (  # noqa: E402
    load_partition_residency_profiles,
    select_fastest_residency,
)
from ragdoll.simulator import generate_poisson_workload  # noqa: E402
from ragdoll.topology import load_topology_profiles, select_fastest_topology  # noqa: E402


def _fixed_batch_selector(batch_size: int) -> Callable[[str, int], int]:
    if batch_size < 1:
        raise ValueError("static_batch_size must be positive")
    return lambda _stage, backlog: min(batch_size, backlog)


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
        "--resident-partitions",
        type=int,
        help="Override the number of Milvus-Lite logical partitions kept resident.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("serial", "static", "adaptive", "profiled"),
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
    if args.resident_partitions is not None:
        if cfg["milvus"].get("mode") != "logical_partitions":
            raise ValueError("--resident-partitions requires milvus.mode=logical_partitions")
        cfg["milvus"]["resident_partitions"] = args.resident_partitions
    elif cfg["milvus"].get("residency_profile"):
        residency_path = Path(cfg["milvus"]["residency_profile"])
        if not residency_path.is_absolute():
            residency_path = PROJECT_ROOT / residency_path
        selected_residency = select_fastest_residency(
            load_partition_residency_profiles(residency_path),
            int(cfg["milvus"]["partition_count"]),
        )
        cfg["milvus"]["resident_partitions"] = selected_residency.resident_partitions
        print(
            "selected resident_partitions="
            f"{selected_residency.resident_partitions} "
            f"profiled_mean_latency={selected_residency.mean_latency_seconds:.4f}"
        )
    questions = [json.loads(line)["question"] for line in Path(cfg["artifacts"]["workload_questions"]).read_text(encoding="utf-8").splitlines()]
    arrivals = generate_poisson_workload(seed=cfg["run"]["seed"], requests_per_phase=cfg["run"]["requests_per_phase"], arrival_rates_per_minute=cfg["run"]["arrival_rates_per_minute"])
    requests = tuple(RAGRequest(item.request_id, questions[item.request_id], item.arrival_time) for item in arrivals)
    profiles = ProfileStore.load(Path(cfg["artifacts"]["profiles"]))
    retriever = build_retriever(cfg)
    generator = build_generator(cfg, PROJECT_ROOT)
    try:
        output: dict[str, object] = {"run": cfg["run"]["name"], "policies": {}}
        for policy in args.policies:
            executed_policy = policy
            selected_topology = None
            if policy == "profiled":
                placement_name = getattr(generator, "placement_name", None)
                if placement_name is None:
                    raise ValueError("profiled topology requires a named memory placement")
                topology_path = Path(cfg["scheduler"]["topology_profile"])
                if not topology_path.is_absolute():
                    topology_path = PROJECT_ROOT / topology_path
                selected_topology = select_fastest_topology(
                    load_topology_profiles(topology_path), placement_name
                )
                executed_policy = selected_topology.topology
                print(
                    f"profiled topology placement={placement_name} "
                    f"selected={executed_policy} "
                    f"profiled_mean_latency={selected_topology.mean_latency_seconds:.4f}"
                )

            if executed_policy == "serial":
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
            elif executed_policy == "static":
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
            summary = _summary(policy, result, time.monotonic() - started)
            summary["executed_policy"] = executed_policy
            placement_name = getattr(generator, "placement_name", None)
            if placement_name is not None:
                summary["placement"] = placement_name
            residency = getattr(retriever, "residency_snapshot", None)
            if residency is not None:
                summary["partition_residency"] = {
                    "resident_partition_ids": list(residency.resident_partition_ids),
                    "loads": residency.loads,
                    "releases": residency.releases,
                    "searches": residency.searches,
                }
                summary["resident_partitions"] = len(residency.resident_partition_ids)
            if selected_topology is not None:
                summary["profiled_mean_latency_seconds"] = selected_topology.mean_latency_seconds
            output["policies"][policy] = summary
            print(f"completed policy={policy} executed_policy={executed_policy}")
        path = args.output or Path(cfg["artifacts"]["result"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
    finally:
        generator.close()
        close_retriever = getattr(retriever, "close", None)
        if close_retriever is not None:
            close_retriever()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
