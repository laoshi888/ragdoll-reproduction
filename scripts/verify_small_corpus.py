"""Read-only integrity check for the Milvus Lite corpus built on AutoDL."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from pymilvus import MilvusClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "real_small.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    client = MilvusClient(cfg["milvus"]["uri"])
    stats = client.get_collection_stats(cfg["milvus"]["collection"])
    client.close()
    print(stats)


if __name__ == "__main__":
    main()
