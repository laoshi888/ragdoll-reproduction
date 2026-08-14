"""Measure real retrieval and generation batch times on AutoDL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragdoll.backends.milvus import MilvusRetriever  # noqa: E402
from ragdoll.backend_factory import build_generator  # noqa: E402
from ragdoll.contracts import ProfileSample, ProfileStore, RAGRequest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "real_small.yaml")
    args = parser.parse_args()
    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    questions = [json.loads(line)["question"] for line in Path(cfg["artifacts"]["workload_questions"]).read_text(encoding="utf-8").splitlines()]
    retriever = MilvusRetriever(uri=cfg["milvus"]["uri"], collection=cfg["milvus"]["collection"], embedder_name=cfg["models"]["embedder"], top_k=cfg["run"]["top_k"])
    generator = build_generator(cfg, PROJECT_ROOT)
    store = ProfileStore()
    candidates = cfg["scheduler"]["generation_batch_candidates"]
    repeats = cfg["scheduler"]["profiling_samples_per_batch"]
    try:
        for size in candidates:
            batch = [RAGRequest(index, question, 0.0) for index, question in enumerate(questions[:size])]
            for _ in range(repeats):
                started = time.monotonic(); retrieved = retriever.retrieve(batch); store.add(ProfileSample("retrieval", size, time.monotonic() - started))
                started = time.monotonic(); generator.generate(retrieved); store.add(ProfileSample("generation", size, time.monotonic() - started))
            print(f"profiled batch_size={size}")
        store.save(Path(cfg["artifacts"]["profiles"]))
    finally:
        generator.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
