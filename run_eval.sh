#!/bin/bash
# Runs the full CORE benchmark (scripts/base_eval.py) over a given HuggingFace model
#
# Usage: HF_TOKEN="your_token" ./run_eval.sh <hf_path> [additional_args...]
# Example: HF_TOKEN="your_token" ./run_eval.sh speakleash/Bielik-1.5B-v3

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable is not set." >&2
  exit 1
fi

if [ $# -eq 0 ]; then
  echo "Error: Please provide a HuggingFace model path as the first argument." >&2
  echo "Usage: $0 <hf_path> [additional_args...]" >&2
  exit 1
fi

HF_PATH="$1"
shift

RUN_NAME=$(basename "$HF_PATH" | tr '[:upper:]' '[:lower:]')

export WANDB_ENTITY="fineweb2edupl"

# Dynamically set the base dir to the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NANOCHAT_BASE_DIR="$SCRIPT_DIR"

cd "$NANOCHAT_BASE_DIR"

# Sync dependencies with uv if uv is available
if command -v uv &> /dev/null; then
  echo "Syncing dependencies with uv..."
  uv sync --extra gpu
fi

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

echo "=== Evaluating ${HF_PATH} ==="
python -m scripts.base_eval \
  --hf-path "${HF_PATH}" \
  --run "${RUN_NAME}" "$@"
