#!/bin/bash

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# --- User Configuration ---
DEPTH="${DEPTH:-20}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:-100}"

# --- HPC Environment Setup ---
# Load required modules (adjust for your HPC system)
module purge
module load cuda/12.1.1 python/3.11.3-gcccore-12.3.0 || true
module list

# Set OpenMP threads based on available CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# GPU count from SLURM
NGPUS=${SLURM_GPUS_ON_NODE:-2}

# --- Directory Structure ---
# Use persistent storage for project base
export NANOCHAT_BASE_DIR="/net/afscra/people/plgmilpo1/test/nanochat-pl"
mkdir -p "$NANOCHAT_BASE_DIR"

# Separate persistent cache from scratch
export UV_CACHE_DIR="${NANOCHAT_BASE_DIR}/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

# Logs directory
LOG_DIR="$NANOCHAT_BASE_DIR/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run_${SLURM_JOB_ID:-local}.log"

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
    uv venv "$VENV_DIR"
fi

log "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Sync dependencies
log "Syncing dependencies..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv sync --extra gpu --locked

# Verify installation
log "Python: $(which python)"
log "Python version: $(python --version)"

# Reset nanochat report
python -m nanochat.report reset

# --- Data Preparation ---
log "=== Preparing training data ==="

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    log "Data directory empty or missing, preparing data..."
    mkdir -p "$DATA_DIR"
    
    # Uncomment and configure your data download method:
    # Option 1: GCS
    # gsutil -m cp "gs://your-bucket/path/*.parquet" "$DATA_DIR/"
    
    # Option 2: Local copy
    # cp /path/to/source/data/*.parquet "$DATA_DIR/"
    
    # Verify data
    if [ -z "$(ls -A $DATA_DIR)" ]; then
        log "ERROR: No data files found after download!"
        exit 1
    fi
    log "Data preparation complete. Files: $(ls -1 $DATA_DIR | wc -l)"
else
    log "Data already exists in $DATA_DIR ($(ls -1 $DATA_DIR | wc -l) files)"
fi

# --- Tokenizer Setup ---
log "=== Setting up custom tokenizer ==="

if [ ! -f "setup_custom_tokenizer.py" ]; then
    log "ERROR: setup_custom_tokenizer.py not found!"
    exit 1
fi

python setup_custom_tokenizer.py 2>&1 | tee -a "$LOGFILE"
log "Tokenizer setup complete"

# --- Pretraining ---
log "=== Starting pretraining ==="
log "Configuration:"
log "  - Depth: $DEPTH"
log "  - Device batch size: $DEVICE_BATCH_SIZE"
log "  - GPUs: $NGPUS"
log "  - Iterations: $NUM_ITERATIONS"

# Run training with error handling
if torchrun \
    --standalone \
    --nproc_per_node="$NGPUS" \
    -m scripts.base_train -- \
    --depth="$DEPTH" \
    --device_batch_size="$DEVICE_BATCH_SIZE" \
    --num_iterations="$NUM_ITERATIONS" \
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