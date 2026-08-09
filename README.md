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

The base Python, PyTorch, CUDA, Transformers, Accelerate, Sentence Transformers, and Datasets environment has been validated on AutoDL. The next step is to identify the paper's public code and its smallest reproducible experiment.
