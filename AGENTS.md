# RAGDoll Reproduction

## Project Scope

- Reproduce the key system behavior described in `paper/RAGDoll.pdf`.
- Start with a small model and dataset. Do not download large models or datasets without checking remote disk capacity.
- Keep the implementation portable between the local workspace and the AutoDL instance.

## Layout

- `src/`: application and experiment code.
- `scripts/`: runnable setup and experiment commands.
- `configs/`: versioned experiment configurations.
- `experiments/`: small, versioned result summaries only.
- `logs/`, `models/`, and `data/`: local or remote generated artifacts; do not commit them.

## Environment

- Remote project path: `/root/autodl-tmp/ragdoll`.
- Conda environment: `ragdoll`.
- Python version: 3.10.
- GPU: NVIDIA RTX 4090, 24 GB VRAM.

## Working Rules

- Explain the relevant paper design before adding a major component.
- Keep each change focused and validate it with the smallest useful test.
- Record executable commands and parameters in `configs/` or `scripts/`.
- Never commit credentials, model weights, datasets, or large raw logs.
