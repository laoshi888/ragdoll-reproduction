"""Run the dependency-free scheduling experiment from a JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragdoll.simulator import (  # noqa: E402
    Policy,
    SimulationConfig,
    StageConfig,
    generate_poisson_workload,
    run_simulation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "simulation_small.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "simulation_small.json")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    retrieval = StageConfig(**{**raw["retrieval"], "batch_candidates": tuple(raw["retrieval"]["batch_candidates"])})
    generation = StageConfig(**{**raw["generation"], "batch_candidates": tuple(raw["generation"]["batch_candidates"])})
    requests = generate_poisson_workload(seed=raw["seed"], **raw["workload"])
    config = SimulationConfig(retrieval=retrieval, generation=generation)
    records = []
    for policy in Policy:
        result = run_simulation(requests, config, policy)
        records.append(
            {
                "policy": policy.value,
                "request_count": len(result.timings),
                "average_latency_seconds": round(result.average_latency_seconds, 6),
                "average_waiting_seconds": round(result.average_waiting_seconds, 6),
                "retrieval_batches": list(result.retrieval_batches),
                "generation_batches": list(result.generation_batches),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(
            f"{record['policy']}: latency={record['average_latency_seconds']:.3f}s "
            f"wait={record['average_waiting_seconds']:.3f}s"
        )


if __name__ == "__main__":
    main()
