"""Build the small, streamed TriviaQA corpus on AutoDL only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ragdoll.corpus import chunk_text, record_contexts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "real_small.yaml")
    parser.add_argument("--reset", action="store_true", help="drop only the configured collection before rebuilding")
    args = parser.parse_args()
    try:
        import yaml
        from datasets import load_dataset
        from pymilvus import MilvusClient
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit("Install requirements-autodl.txt in AutoDL before building the corpus.") from error
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    corpus, run, artifacts = config["corpus"], config["run"], config["artifacts"]
    client = MilvusClient(config["milvus"]["uri"])
    collection = config["milvus"]["collection"]
    if client.has_collection(collection):
        if not args.reset:
            raise SystemExit(f"Collection {collection!r} already exists; rerun with --reset to replace it.")
        client.drop_collection(collection)
    model_cache = Path(artifacts["model_cache"])
    data_cache = Path(artifacts["data_cache"])
    model_cache.mkdir(parents=True, exist_ok=True)
    data_cache.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer(config["models"]["embedder"], cache_folder=str(model_cache))
    dimension = embedder.get_embedding_dimension()
    client.create_collection(collection_name=collection, dimension=dimension, metric_type="COSINE", auto_id=False)
    dataset = load_dataset(
        corpus["dataset"],
        corpus["dataset_config"],
        split=corpus["split"],
        streaming=corpus["streaming"],
        cache_dir=str(data_cache),
    )
    max_questions = run["requests_per_phase"] * len(run["arrival_rates_per_minute"])
    texts: list[str] = []
    questions: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in dataset:
        question = record.get("question")
        if isinstance(question, str) and len(questions) < max_questions:
            questions.append({"question": question})
        for context in record_contexts(record):
            for piece in chunk_text(context, chunk_size=corpus["chunk_size_characters"], overlap=corpus["chunk_overlap_characters"]):
                if len(texts) >= corpus["max_chunks"]:
                    break
                if piece not in seen:
                    seen.add(piece)
                    texts.append(piece)
                if len(texts) >= corpus["max_chunks"]:
                    break
            if len(texts) >= corpus["max_chunks"]:
                break
        if len(texts) >= corpus["max_chunks"] and len(questions) >= max_questions:
            break
    if not texts or len(questions) < max_questions:
        raise SystemExit("Stream ended before enough contexts or workload questions were collected.")
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    client.insert(collection_name=collection, data=[{"id": index, "vector": vector.tolist(), "text": text} for index, (text, vector) in enumerate(zip(texts, vectors, strict=True))])
    question_path = Path(artifacts["workload_questions"])
    question_path.parent.mkdir(parents=True, exist_ok=True)
    question_path.write_text("".join(json.dumps(item) + "\n" for item in questions), encoding="utf-8")
    client.close()
    print(f"collection={collection} chunks={len(texts)} questions={len(questions)}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
