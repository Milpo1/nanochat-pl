#!/bin/bash

# This script runs a pretraining-only pipeline with a custom HF tokenizer and custom data.
# It skips tokenizer training, midtraining, and SFT.

# --- User Configuration ---
# 1. Set your GCS bucket and data path.
#    Assumes your data is in parquet format with a 'text' column.
GCS_DATA_PATH="gs://fineweb2-pl/huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/pol_Latn/train/000_00000.parquet"

# 2. Set your WandB run name for logging, or "dummy" to disable.
WANDB_RUN="custom-pretrain-run"
# --- End User Configuration ---

set -e # Exit immediately if a command exits with a non-zero status.

# 1. Standard nanochat setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
python -m nanochat.report reset

# 2. Prepare Custom Data
# Download your pretraining data from GCS to the directory nanochat expects.
DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
echo "Preparing custom training data..."
mkdir -p $DATA_DIR
# Make sure you have gcloud CLI installed and authenticated (gcloud auth application-default login)
echo "Downloading data from GCS to $DATA_DIR..."
gsutil -m cp "$GCS_DATA_PATH" "$DATA_DIR/"
echo "Data download complete."

# 3. Prepare Custom Tokenizer
# This runs the script you created in Step 2.
echo "Setting up custom Hugging Face tokenizer..."
python setup_custom_tokenizer.py
echo "Tokenizer setup complete."

# 4. Download Evaluation Data
# The `base_eval.py` script requires this bundle.
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "Downloading eval_bundle..."
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# 5. Run Pretraining
echo "Starting pretraining..."
# IMPORTANT: Adjust --depth and --device_batch_size for your specific model and hardware.
# Using a different tokenizer changes the vocab size, which changes the total parameter count.
# This affects memory usage.
#
# How to tune:
# - Start with a low --device_batch_size (e.g., 4 or 8).
# - Run the script. If it runs out of memory (OOM), decrease it.
# - If it runs fine, you can try increasing it to improve GPU utilization (MFU).
# - Refer to the detailed comments in `run1000.sh` for an example of this tuning process.
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=16 \
    --run=$WANDB_RUN
    --num_iterations=3000

# 6. Evaluate Pretrained Model (Optional but Recommended)
echo "Evaluating pretrained model..."
torchrun --standalone --nproc_per_node=4 -m scripts.base_loss
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval

# 7. Generate Final Report
python -m nanochat.report generate

echo "Custom pretraining run finished!"