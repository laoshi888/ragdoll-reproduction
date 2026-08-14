"""Profile logical-partition residency counts using isolated real pipeline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "real_flex_partitioned.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "experiments" / "flex_partitioned_residency_profile.json"
    )
    args = parser.parse_args()
    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    scheduler = cfg["scheduler"]
    candidates = tuple(int(value) for value in scheduler["partition_residency_candidates"])
    repeats = int(scheduler["partition_residency_profile_repeats"])
    partition_count = int(cfg["milvus"]["partition_count"])
    if not candidates or repeats < 1:
        raise SystemExit("partition residency candidates and repeats must be positive")
    if any(value < 1 or value > partition_count for value in candidates):
        raise SystemExit("partition residency candidate is outside the configured partition range")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.parent / "partition_residency_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        # Rotate candidate order so first-run cache and allocator effects do not
        # systematically favor a particular residency count.
        order = candidates[repeat:] + candidates[:repeat]
        for resident_count in order:
            raw_path = raw_dir / f"resident_{resident_count}_repeat_{repeat}.json"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_real_pipeline.py"),
                "--config",
                str(args.config),
                "--resident-partitions",
                str(resident_count),
                "--policies",
                "serial",
                "--output",
                str(raw_path),
            ]
            print(f"profiling resident_partitions={resident_count} repeat={repeat + 1}/{repeats}")
            subprocess.run(command, check=True)
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            metrics = payload["policies"]["serial"]
            rows.append(
                {
                    "resident_partitions": resident_count,
                    "repeat": repeat,
                    "mean_latency_seconds": metrics["mean_latency_seconds"],
                    "p95_latency_seconds": metrics["p95_latency_seconds"],
                    "mean_retrieval_seconds": metrics["mean_retrieval_seconds"],
                    "loads": metrics["partition_residency"]["loads"],
                    "releases": metrics["partition_residency"]["releases"],
                }
            )

    summary: list[dict[str, object]] = []
    for resident_count in candidates:
        matching = [row for row in rows if row["resident_partitions"] == resident_count]
        summary.append(
            {
                "resident_partitions": resident_count,
                "runs": len(matching),
                "mean_latency_seconds": sum(float(row["mean_latency_seconds"]) for row in matching) / len(matching),
                "mean_retrieval_seconds": sum(float(row["mean_retrieval_seconds"]) for row in matching) / len(matching),
                "mean_loads": sum(float(row["loads"]) for row in matching) / len(matching),
                "mean_releases": sum(float(row["releases"]) for row in matching) / len(matching),
            }
        )
    selected = min(summary, key=lambda row: float(row["mean_latency_seconds"]))
    output = {"runs": rows, "summary": summary, "selected": selected}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
