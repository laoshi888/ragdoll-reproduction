"""Profile logical or native partition residency with isolated real runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _memory_gib(value: str) -> float:
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]?i?B)\s*", value)
    if not match:
        raise ValueError(f"unsupported Docker memory value: {value!r}")
    number = float(match.group(1))
    factors = {
        "B": 1 / 2**30,
        "KB": 1000 / 2**30,
        "KiB": 1 / 2**20,
        "MB": 1000**2 / 2**30,
        "MiB": 1 / 1024,
        "GB": 1000**3 / 2**30,
        "GiB": 1.0,
        "TB": 1000**4 / 2**30,
        "TiB": 1024.0,
    }
    return number * factors[match.group(2)]


def _docker_memory_snapshot(container: str) -> dict[str, object]:
    inspect_output = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}} {{.State.OOMKilled}} "
            "{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}",
            container,
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip().split()
    running = inspect_output[0].lower() == "true"
    snapshot: dict[str, object] = {
        "running": running,
        "oom_killed": inspect_output[1].lower() == "true",
        "limit_bytes": int(inspect_output[2]),
        "memory_swap_bytes": int(inspect_output[3]),
        "usage_gib": None,
        "displayed_limit_gib": None,
    }
    if running:
        usage_output = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        usage_text, limit_text = (item.strip() for item in usage_output.split("/", 1))
        snapshot["usage_gib"] = _memory_gib(usage_text)
        snapshot["displayed_limit_gib"] = _memory_gib(limit_text)
    return snapshot


def _recover_container(container: str, timeout_seconds: float = 180.0) -> None:
    state = _docker_memory_snapshot(container)
    if not state["running"]:
        subprocess.run(["docker", "start", container], check=True)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        health = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}",
                container,
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if health in {"healthy", "running"}:
            return
        time.sleep(2)
    raise RuntimeError(f"container {container!r} did not recover within {timeout_seconds}s")


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
    selection_metric = scheduler.get(
        "partition_residency_selection_metric", "mean_latency_seconds"
    )
    if selection_metric not in {"mean_latency_seconds", "median_latency_seconds"}:
        raise SystemExit("unsupported partition residency selection metric")
    partition_count = int(cfg["milvus"]["partition_count"])
    if not candidates or repeats < 1:
        raise SystemExit("partition residency candidates and repeats must be positive")
    if any(value < 1 or value > partition_count for value in candidates):
        raise SystemExit("partition residency candidate is outside the configured partition range")
    memory_cfg = cfg.get("memory")
    if memory_cfg:
        host_budget = float(memory_cfg["host_budget_gib"])
        llm_reserve = float(memory_cfg["llm_host_reserve_gib"])
        configured_limit = float(memory_cfg["milvus_container_limit_gib"])
        if min(host_budget, llm_reserve, configured_limit) <= 0:
            raise SystemExit("memory budgets must be positive")
        if llm_reserve + configured_limit > host_budget:
            raise SystemExit("LLM reserve plus Milvus limit exceeds the host budget")
        initial_memory = _docker_memory_snapshot(memory_cfg["milvus_container"])
        expected_limit_bytes = round(configured_limit * 2**30)
        if int(initial_memory["limit_bytes"]) != expected_limit_bytes:
            raise SystemExit(
                "Milvus cgroup limit does not match config: "
                f"actual={initial_memory['limit_bytes']} expected={expected_limit_bytes}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.parent / f"{args.output.stem}_runs"
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
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                if not memory_cfg:
                    raise
                try:
                    memory = _docker_memory_snapshot(memory_cfg["milvus_container"])
                except (OSError, ValueError, subprocess.CalledProcessError) as snapshot_error:
                    memory = {"snapshot_error": str(snapshot_error)}
                rows.append(
                    {
                        "resident_partitions": resident_count,
                        "repeat": repeat,
                        "feasible": False,
                        "error": f"pipeline exited with status {error.returncode}",
                        "milvus_memory": memory,
                    }
                )
                try:
                    _recover_container(memory_cfg["milvus_container"])
                except (OSError, RuntimeError, subprocess.CalledProcessError) as recovery_error:
                    raise SystemExit(
                        f"Milvus failed and could not recover: {recovery_error}"
                    ) from recovery_error
                continue
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            metrics = payload["policies"]["serial"]
            row = {
                "resident_partitions": resident_count,
                "repeat": repeat,
                "mean_latency_seconds": metrics["mean_latency_seconds"],
                "p95_latency_seconds": metrics["p95_latency_seconds"],
                "mean_retrieval_seconds": metrics["mean_retrieval_seconds"],
                "loads": metrics["partition_residency"]["loads"],
                "releases": metrics["partition_residency"]["releases"],
                "feasible": True,
            }
            if memory_cfg:
                memory = _docker_memory_snapshot(memory_cfg["milvus_container"])
                if memory["usage_gib"] is None:
                    raise SystemExit("Milvus stopped before its memory usage could be measured")
                process_rss = metrics.get("process_max_rss_gib")
                total_observed = (
                    float(memory["usage_gib"]) + float(process_rss)
                    if process_rss is not None
                    else None
                )
                row.update(
                    {
                        "process_max_rss_gib": process_rss,
                        "milvus_memory": memory,
                        "observed_total_host_gib": total_observed,
                        "host_budget_gib": float(memory_cfg["host_budget_gib"]),
                    }
                )
                row["feasible"] = bool(
                    not memory["oom_killed"]
                    and total_observed is not None
                    and total_observed <= float(memory_cfg["host_budget_gib"])
                )
            rows.append(row)

    summary: list[dict[str, object]] = []
    for resident_count in candidates:
        matching = [
            row
            for row in rows
            if row["resident_partitions"] == resident_count and row.get("feasible", True)
        ]
        if not matching:
            continue
        summary_row = {
            "resident_partitions": resident_count,
            "runs": len(matching),
            "mean_latency_seconds": sum(float(row["mean_latency_seconds"]) for row in matching) / len(matching),
            "median_latency_seconds": statistics.median(
                float(row["mean_latency_seconds"]) for row in matching
            ),
            "mean_retrieval_seconds": sum(float(row["mean_retrieval_seconds"]) for row in matching) / len(matching),
            "mean_loads": sum(float(row["loads"]) for row in matching) / len(matching),
            "mean_releases": sum(float(row["releases"]) for row in matching) / len(matching),
        }
        if memory_cfg:
            summary_row.update(
                {
                    "mean_milvus_memory_gib": sum(
                        float(row["milvus_memory"]["usage_gib"]) for row in matching
                    )
                    / len(matching),
                    "max_observed_total_host_gib": max(
                        float(row["observed_total_host_gib"]) for row in matching
                    ),
                }
            )
        summary.append(summary_row)
    if not summary:
        raise SystemExit("No feasible partition residency configuration completed")
    selected = min(summary, key=lambda row: float(row[selection_metric]))
    output = {
        "memory_constraint": memory_cfg,
        "selection_metric": selection_metric,
        "runs": rows,
        "summary": summary,
        "infeasible_runs": [row for row in rows if not row.get("feasible", True)],
        "selected": selected,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
