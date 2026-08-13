"""Run the profiled two-worker RAG pipeline on AutoDL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragdoll.backends.milvus import MilvusRetriever  # noqa: E402
from ragdoll.backends.vllm import VLLMGenerator  # noqa: E402
from ragdoll.batching import ProfiledBatchSelector  # noqa: E402
from ragdoll.contracts import ProfileStore, RAGRequest  # noqa: E402
from ragdoll.runtime import PipelinedRAGRunner  # noqa: E402
from ragdoll.simulator import generate_poisson_workload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "real_small.yaml")
    args = parser.parse_args()
    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    questions = [json.loads(line)["question"] for line in Path(cfg["artifacts"]["workload_questions"]).read_text(encoding="utf-8").splitlines()]
    arrivals = generate_poisson_workload(seed=cfg["run"]["seed"], requests_per_phase=cfg["run"]["requests_per_phase"], arrival_rates_per_minute=cfg["run"]["arrival_rates_per_minute"])
    requests = tuple(RAGRequest(item.request_id, questions[item.request_id], item.arrival_time) for item in arrivals)
    profiles = ProfileStore.load(Path(cfg["artifacts"]["profiles"]))
    retriever = MilvusRetriever(uri=cfg["milvus"]["uri"], collection=cfg["milvus"]["collection"], embedder_name=cfg["models"]["embedder"], top_k=cfg["run"]["top_k"])
    generator = VLLMGenerator(
        model=cfg["models"]["generator"],
        max_new_tokens=cfg["run"]["max_new_tokens"],
        **cfg["vllm"],
    )
    selector = ProfiledBatchSelector(tuple(cfg["scheduler"]["generation_batch_candidates"]), profiles, cfg["scheduler"]["static_batch_size"])
    result = PipelinedRAGRunner(retriever=retriever, generator=generator, retrieval_selector=selector, generation_selector=selector).run(requests)
    output = {"request_count": len(result.responses), "mean_latency_seconds": sum(item.latency_seconds for item in result.timings) / len(result.timings), "retrieval_batches": result.retrieval_batch_sizes, "generation_batches": result.generation_batch_sizes}
    path = Path(cfg["artifacts"]["result"]); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
