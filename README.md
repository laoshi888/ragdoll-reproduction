# RAGDoll Reproduction

This repository is a reproduction workspace for *RAGDoll: Efficient Offloading-based Online RAG System on a Single GPU*.

## Current Environment

- Local development: `D:\paper`
- Remote testing: AutoDL RTX 4090, Python 3.10 Conda environment `ragdoll`
- Remote project path: `/root/autodl-tmp/ragdoll`

## Workflow

1. Develop and review code locally.
2. Commit only source code, scripts, and configuration files.
3. Pull the commit on AutoDL and run GPU experiments there.
4. Keep models, datasets, and large logs out of Git.

## Status

The small-scale reproduction now covers FlexLLMGen GPU/CPU placement, profiled
batch and topology selection, Milvus-Lite logical partition residency, and
joint end-to-end configuration selection.  A 16,000-vector pressure profile
confirmed measurable lazy load/release overhead.  The active final stage adds
Milvus Standalone native partitions under a real Docker cgroup memory limit;
see `experiments/native_partition_memory_design.md` and
`scripts/run_autodl_native_partition_profile.sh`.

## External Components

This reproduction reuses established open-source components where possible:

- FlexLLMGen (the official FlexGen implementation): https://github.com/FMInference/FlexLLMGen
- Milvus vector database: https://github.com/milvus-io/milvus
- vLLM baseline serving engine: https://github.com/vllm-project/vllm
- Hugging Face Transformers and Accelerate for model interfaces and device placement.

Pinned external versions and roles are recorded in `configs/sources.yaml`.
