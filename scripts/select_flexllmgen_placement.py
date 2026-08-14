"""Choose a measured FlexLLMGen placement for a GPU-memory budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragdoll.placement import PlacementProfile, select_fastest_feasible  # noqa: E402


def load_profiles(path: Path) -> tuple[PlacementProfile, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        PlacementProfile(
            name=item["name"],
            percent=tuple(item["percent"]),
            peak_gpu_memory_gib=item["peak_gpu_memory_gib"],
            total_latency_seconds=item["total_latency_seconds"],
            decode_throughput_tokens_per_second=item["decode_throughput_tokens_per_second"],
        )
        for item in raw["placements"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "flexllmgen_opt13b_profile.json",
    )
    parser.add_argument("--max-gpu-memory-gib", type=float, required=True)
    args = parser.parse_args()

    selected = select_fastest_feasible(load_profiles(args.config), args.max_gpu_memory_gib)
    print(
        json.dumps(
            {
                "name": selected.name,
                "percent": list(selected.percent),
                "peak_gpu_memory_gib": selected.peak_gpu_memory_gib,
                "total_latency_seconds": selected.total_latency_seconds,
                "decode_throughput_tokens_per_second": selected.decode_throughput_tokens_per_second,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
