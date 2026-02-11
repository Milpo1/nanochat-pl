#!/bin/env bash
module purge
module load CUDA/12.8.0 ML-bundle/25.04

source .venv/bin/activate

export NANOCHAT_BASE_DIR="$(pwd)"
export UV_CACHE_DIR="${NANOCHAT_BASE_DIR}/.cache/uv"
UV_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL" UV_CACHE_DIR="$UV_CACHE_DIR" uv sync --extra gpu

uv run python -m scripts.eval_all_checkpoints