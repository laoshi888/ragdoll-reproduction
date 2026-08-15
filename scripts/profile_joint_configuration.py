"""Measure complete placement, residency, and topology configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "real_flex_partitioned.yaml"
    )
    parser.add_argument(
        "--plan", type=Path, default=PROJECT_ROOT / "configs" / "flex_joint_profile_plan.json"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "experiments" / "flex_joint_profile.json"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing isolated run files and execute only missing repetitions",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    candidates = tuple(plan["candidates"])
    repeats = int(plan["repeats"])
    if not candidates or repeats < 1:
        raise SystemExit("joint profile plan requires candidates and positive repeats")
    names = [str(candidate["name"]) for candidate in candidates]
    if len(set(names)) != len(names):
        raise SystemExit("joint profile candidate names must be unique")
    if any(float(candidate["max_gpu_memory_gib"]) <= 0 for candidate in candidates):
        raise SystemExit("joint profile GPU budgets must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.parent / "joint_configuration_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        order = candidates[repeat:] + candidates[:repeat]
        for candidate in order:
            name = str(candidate["name"])
            budget = float(candidate["max_gpu_memory_gib"])
            raw_path = raw_dir / f"{name}_repeat_{repeat}.json"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_real_pipeline.py"),
                "--config",
                str(args.config),
                "--max-gpu-memory-gib",
                str(budget),
                "--policies",
                "profiled",
                "--output",
                str(raw_path),
            ]
            if args.resume and raw_path.exists():
                print(f"reusing joint_candidate={name} repeat={repeat + 1}/{repeats}")
            else:
                print(f"profiling joint_candidate={name} repeat={repeat + 1}/{repeats}")
                subprocess.run(command, check=True)
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            metrics = payload["policies"]["profiled"]
            rows.append(
                {
                    "name": name,
                    "repeat": repeat,
                    "max_gpu_memory_gib": budget,
                    "placement": metrics["placement"],
                    "resident_partitions": metrics["resident_partitions"],
                    "topology": metrics["executed_policy"],
                    "mean_latency_seconds": metrics["mean_latency_seconds"],
                    "p95_latency_seconds": metrics["p95_latency_seconds"],
                    "mean_waiting_seconds": metrics["mean_waiting_seconds"],
                    "mean_retrieval_seconds": metrics["mean_retrieval_seconds"],
                    "mean_generation_seconds": metrics["mean_generation_seconds"],
                    "wall_seconds": metrics["wall_seconds"],
                }
            )

    summary: list[dict[str, object]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        matching = [row for row in rows if row["name"] == name]
        first = matching[0]
        summary.append(
            {
                "name": name,
                "runs": len(matching),
                "max_gpu_memory_gib": float(candidate["max_gpu_memory_gib"]),
                "placement": first["placement"],
                "resident_partitions": first["resident_partitions"],
                "topology": first["topology"],
                "mean_latency_seconds": sum(float(row["mean_latency_seconds"]) for row in matching) / len(matching),
                "median_latency_seconds": median(float(row["mean_latency_seconds"]) for row in matching),
                "mean_waiting_seconds": sum(float(row["mean_waiting_seconds"]) for row in matching) / len(matching),
                "median_waiting_seconds": median(float(row["mean_waiting_seconds"]) for row in matching),
                "mean_retrieval_seconds": sum(float(row["mean_retrieval_seconds"]) for row in matching) / len(matching),
                "median_retrieval_seconds": median(float(row["mean_retrieval_seconds"]) for row in matching),
                "mean_generation_seconds": sum(float(row["mean_generation_seconds"]) for row in matching) / len(matching),
                "median_generation_seconds": median(float(row["mean_generation_seconds"]) for row in matching),
                "mean_wall_seconds": sum(float(row["wall_seconds"]) for row in matching) / len(matching),
                "median_wall_seconds": median(float(row["wall_seconds"]) for row in matching),
            }
        )
    output = {"runs": rows, "summary": summary}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
