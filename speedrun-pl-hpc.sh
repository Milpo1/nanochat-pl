#!/bin/env bash

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# --- User Configuration ---
DEPTH="${DEPTH:-20}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:-100}"

WANDB_RUN="${WANDB_RUN:-fineweb2edupl-hpc-$(date +%Y%m%d-%H%M)}"
# --- HPC Environment Setup ---
# Load required modules (adjust for your HPC system)
module purge

module load CUDA/12.8.0 ML-bundle/25.04

module list

# Set OpenMP threads based on available CPUs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=$OMP_NUM_THREADS


# This forces the CPU to wait for every single GPU kernel. 
# If a kernel fails (due to NaN or logic), Python crashes INSTANTLY at that line.
export CUDA_LAUNCH_BLOCKING=1

# 2. DEBUG DISTRIBUTED PROCESSES
# Logs every collective operation (Start/End). You will see exactly which rank stops reporting.
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# 3. DEBUG NCCL (The Communications)
# Shows if NCCL receives invalid arguments or breaks connection.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 4. FLIGHT RECORDER (Kernel Trace)
# Keeps a buffer of the last 20k kernels so if it hangs, we can dump the state.
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_FR_BUFFER_SIZE=20000

# GPU count from SLURM
NGPUS=${SLURM_GPUS_ON_NODE:-1}

export UV_CACHE_DIR="${NANOCHAT_BASE_DIR}/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# Logs directory
LOG_DIR="$NANOCHAT_BASE_DIR/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run_${SLURM_JOB_ID:-local}.log"

export WANDB_DIR="$NANOCHAT_BASE_DIR/wandb_logs"
mkdir -p "$WANDB_DIR"
export WANDB_PROJECT="fineweb2edupl"

# Virtual environment in persistent location
VENV_DIR="$NANOCHAT_BASE_DIR/.venv"

# Data and checkpoint directories
DATA_DIR="$NANOCHAT_BASE_DIR/base_data"

# --- Logging Function ---
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

# --- System Information ---
log "=== Job Information ==="
log "Job ID: ${SLURM_JOB_ID:-N/A}"
log "Node: $(hostname)"
log "GPUs: $NGPUS"
log "CPUs per task: ${SLURM_CPUS_PER_TASK:-N/A}"
log "OMP Threads: $OMP_NUM_THREADS"
log "Working Directory: $(pwd)"
log "Base Directory: $NANOCHAT_BASE_DIR"

if command -v nvidia-smi &> /dev/null; then
    log "=== GPU Information ==="
    nvidia-smi | tee -a "$LOGFILE"
fi

# --- UV and Environment Setup ---
log "=== Setting up Python environment ==="

# Install uv if not available
if ! command -v uv &> /dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create or activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment at $VENV_DIR..."
    uv venv "$VENV_DIR" --python $(which python)
fi

log "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Sync dependencies
log "Syncing dependencies..."
UV_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL" UV_CACHE_DIR="$UV_CACHE_DIR" uv sync --extra gpu

# Verify installation
log "Python: $(which python)"
log "Python version: $(python --version)"

log "$(python -c "import torch; print(f'Torch Version: {torch.__version__}\nCUDA Available: {torch.cuda.is_available()}\nDevice Name: {torch.cuda.get_device_name(0)}')")"
# Reset nanochat report
python -m nanochat.report reset

# --- Data Preparation ---
log "=== Preparing training data ==="

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    log "Data directory empty or missing, preparing data..."
    mkdir -p "$DATA_DIR"
    uv pip install google-cloud-storage
    python gcp_fetch.py --key=$HOME/.keys/gcs-read-only.json --src=$NANOCHAT_DATA_SOURCE_PATTERN --dest=$DATA_DIR
    
    # Verify data
    if [ -z "$(ls -A $DATA_DIR)" ]; then
        log "ERROR: No data files found after download!"
        exit 1
    fi
    log "Data preparation complete. Files: $(ls -1 $DATA_DIR | wc -l)"
else
    log "Data already exists in $DATA_DIR ($(ls -1 $DATA_DIR | wc -l) files)"
fi

# # --- Tokenizer Setup ---
# log "=== Setting up custom tokenizer ==="

# if [ ! -f "setup_custom_tokenizer.py" ]; then
#     log "ERROR: setup_custom_tokenizer.py not found!"
#     exit 1
# fi

# python setup_custom_tokenizer.py 2>&1 | tee -a "$LOGFILE"
# log "Tokenizer setup complete"

# --- Pretraining ---
log "=== Starting pretraining ==="
log "Configuration:"
log "  - Depth: $DEPTH"
log "  - Device batch size: $DEVICE_BATCH_SIZE"
log "  - GPUs: $NGPUS"
log "  - Iterations: $NUM_ITERATIONS"
log "  - WandB Run: $WANDB_RUN"


export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_DISTRIBUTED_DEBUG=DETAIL


# Run training with error handling
if torchrun \
    --standalone \
    --log-dir=logs \
    --redirects 3 \
    --tee 3 \
    --nproc_per_node="$NGPUS" \
    -m scripts.base_train -- \
    --depth="$DEPTH" \
    --device_batch_size="$DEVICE_BATCH_SIZE" \
    --num_iterations="$NUM_ITERATIONS" \
    --eval_every="$EVAL_EVERY" \
    --core_metric_every="$CORE_METRIC_EVERY" \
    --save_every="$SAVE_EVERY" \
    --run="$WANDB_RUN" \
    --resume_from_step=38000 \
    2>&1 | tee -a "$LOGFILE"; then
    log "Pretraining completed successfully!"
else
    log "ERROR: Pretraining failed with exit code $?"
    exit 1
fi

# --- Optional Evaluation ---
# Uncomment to enable evaluation after training
# log "=== Evaluating model ==="
# torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.base_loss 2>&1 | tee -a "$LOGFILE"
# torchrun --standalone --nproc_per_node="$NGPUS" -m scripts.base_eval 2>&1 | tee -a "$LOGFILE"

# --- Report Generation ---
# log "=== Generating report ==="
# python -m nanochat.report generate 2>&1 | tee -a "$LOGFILE"

# --- Completion ---
log "=== Pipeline Complete ==="
log "Log file: $LOGFILE"
log "Total runtime: $SECONDS seconds"

exit 0
