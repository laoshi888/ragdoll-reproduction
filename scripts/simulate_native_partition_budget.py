"""Evaluate a transparent, dependency-free residency-budget model.

This is an analytical simulation calibrated with the already measured logical
partition-pressure latencies. It does not start Milvus or claim a physical
memory measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate(raw: dict[str, object]) -> dict[str, object]:
    budget = float(raw["host_memory_budget_gib"])
    reserve = float(raw["llm_process_reserve_gib"])
    footprint = float(raw["resident_partition_footprint_gib"])
    latencies = {int(key): float(value) for key, value in raw["observed_end_to_end_latency_seconds"].items()}
    candidates = [int(value) for value in raw["candidate_resident_partitions"]]

    if budget <= 0 or reserve < 0 or footprint <= 0:
        raise ValueError("budget and partition footprint must be positive; reserve cannot be negative")
    if reserve > budget:
        raise ValueError("LLM reserve exceeds the simulated host-memory budget")
    if any(candidate <= 0 or candidate not in latencies for candidate in candidates):
        raise ValueError("each candidate must be positive and have a latency observation")

    rows = []
    for resident_count in candidates:
        estimated_usage = reserve + resident_count * footprint
        rows.append(
            {
                "resident_partitions": resident_count,
                "estimated_total_memory_gib": round(estimated_usage, 3),
                "within_budget": estimated_usage <= budget,
                "calibrated_latency_seconds": latencies[resident_count],
            }
        )
    feasible = [row for row in rows if row["within_budget"]]
    if not feasible:
        raise ValueError("no residency candidate fits the simulated budget")
    selected = min(feasible, key=lambda row: (row["calibrated_latency_seconds"], row["resident_partitions"]))
    return {
        "simulation_label": raw["simulation_label"],
        "model": "LLM reserve + resident partitions * per-partition footprint <= budget",
        "inputs": {
            "host_memory_budget_gib": budget,
            "llm_process_reserve_gib": reserve,
            "resident_partition_footprint_gib": footprint,
            "latency_source": raw["source_experiment"],
        },
        "candidates": rows,
        "selected_candidate": selected,
        "limitation": "Estimated footprints are assumptions; this does not validate native Milvus residency or cgroup memory use.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "simulated_native_partition_budget.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "simulated_native_partition_budget.json")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    selected = result["selected_candidate"]
    print(
        "selected_resident_partitions="
        f"{selected['resident_partitions']} "
        f"estimated_memory_gib={selected['estimated_total_memory_gib']} "
        f"latency_seconds={selected['calibrated_latency_seconds']}"
    )


if __name__ == "__main__":
    main()
