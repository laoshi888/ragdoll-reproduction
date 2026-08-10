"""Read-only environment preflight. Run this on AutoDL before downloading assets."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    project = Path("/root/autodl-tmp/ragdoll")
    storage_root = project if project.exists() else project.parent
    if not storage_root.exists():
        print(f"AutoDL storage path not present: {storage_root}")
        print("Run this preflight on the AutoDL Linux instance before downloading assets.")
        return
    usage = shutil.disk_usage(storage_root)
    print(f"python={sys.version.split()[0]}")
    print(f"disk_free_gib={usage.free / 2**30:.2f}")
    for package in ("torch", "transformers", "sentence_transformers", "datasets", "pymilvus", "vllm"):
        print(f"{package}={'installed' if importlib.util.find_spec(package) else 'missing'}")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True, text=True, capture_output=True)
        print(f"gpu={result.stdout.strip()}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("gpu=nvidia-smi unavailable")


if __name__ == "__main__":
    main()
