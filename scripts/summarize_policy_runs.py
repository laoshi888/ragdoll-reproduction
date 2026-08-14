"""Summarize repeated real-pipeline runs written by the batch experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


METRICS = (
    "mean_latency_seconds",
    "p95_latency_seconds",
    "mean_waiting_seconds",
    "mean_retrieval_seconds",
    "mean_generation_seconds",
    "wall_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values: dict[str, dict[str, list[float]]] = {}
    for path in sorted(args.input_dir.glob("round*_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for policy, result in raw["policies"].items():
            policy_values = values.setdefault(policy, {metric: [] for metric in METRICS})
            for metric in METRICS:
                policy_values[metric].append(result[metric])
    if not values:
        raise ValueError(f"no round result files found in {args.input_dir}")

    summary = {
        policy: {
            "runs": len(policy_values["mean_latency_seconds"]),
            "mean": {
                metric: statistics.mean(metric_values)
                for metric, metric_values in policy_values.items()
            },
            "median": {
                metric: statistics.median(metric_values)
                for metric, metric_values in policy_values.items()
            },
        }
        for policy, policy_values in sorted(values.items())
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
