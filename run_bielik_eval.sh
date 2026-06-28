#!/bin/bash
# Runs the full CORE benchmark (scripts/base_eval.py) over speakleash/Bielik-1.5B-v3
#
# Usage: HF_TOKEN="your_token" ./run_bielik_eval.sh

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable is not set." >&2
  exit 1
fi

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

echo "=== Evaluating Bielik-1.5B-v3 ==="
python -m scripts.base_eval \
  --hf-path "speakleash/Bielik-1.5B-v3" \
  --run "bielik-1.5b-v3"
