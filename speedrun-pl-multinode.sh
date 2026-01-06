#!/bin/env bash

set -euo pipefail

DEPTH="${DEPTH:-32}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-68500}"
WANDB_RUN="${WANDB_RUN:-fineweb2edupl-multinode-run-$(date +%Y%m%d-%H%M)}"

# SLURM_NODEID is 0 for the first node, 1 for the second, etc.
export NODE_RANK=$SLURM_NODEID
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-4}

export UV_CACHE_DIR="${NANOCHAT_BASE_DIR}/.cache/uv"
mkdir -p "$UV_CACHE_DIR"
mkdir -p "${NANOCHAT_BASE_DIR}/base_checkpoints/d${DEPTH}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

export NCCL_SOCKET_IFNAME=hsn

# Log file specific to this node
LOG_DIR="$NANOCHAT_BASE_DIR/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run_${SLURM_JOB_ID}_node_${NODE_RANK}.log"

export WANDB_DIR="$NANOCHAT_BASE_DIR/wandb_logs"
mkdir -p "$WANDB_DIR"
export WANDB_PROJECT="fineweb2edupl"

log() {
    echo "[Node $NODE_RANK] [$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "Starting on $(hostname)"
log "Role: Node $NODE_RANK of $NNODES"
log "Master: $MASTER_ADDR:$MASTER_PORT"

VENV_DIR="$NANOCHAT_BASE_DIR/.venv"
SETUP_FLAG="$NANOCHAT_BASE_DIR/.setup_complete"

if [ "$NODE_RANK" -eq 0 ]; then

        # --- System Information ---
    log "=== Job Information ==="
    log "Job ID: ${SLURM_JOB_ID:-N/A}"
    log "Node: $(hostname)"
    log "GPUs per node: $NPROC_PER_NODE"
    log "Number of nodes: $NNODES"
    log "CPUs per task: ${SLURM_CPUS_PER_TASK:-N/A}"
    log "OMP Threads: $OMP_NUM_THREADS"
    log "Working Directory: $(pwd)"
    log "Base Directory: $NANOCHAT_BASE_DIR"

    if command -v nvidia-smi &> /dev/null; then
        log "=== GPU Information Node: $(hostname) ==="
        nvidia-smi | tee -a "$LOGFILE"
    fi

    log "=== Node 0: Initializing Environment ==="

    if ! command -v uv &> /dev/null; then
        log "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    if [ ! -d "$VENV_DIR" ]; then
        log "Creating venv..."
        uv venv "$VENV_DIR" --python $(which python)
    fi

    source "$VENV_DIR/bin/activate"
    UV_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL" UV_CACHE_DIR="$UV_CACHE_DIR" uv sync --extra gpu
    log "Python: $(which python)"
    log "Python version: $(python --version)"
    log "$(python -c "import torch; print(f'Torch Version: {torch.__version__}\nCUDA Available: {torch.cuda.is_available()}\nDevice Name: {torch.cuda.get_device_name(0)}')")"

    # Only Node 0 downloads
    DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
        log "Downloading data..."
        uv pip install google-cloud-storage
        python gcp_fetch.py --key=$HOME/.keys/gcs-read-only.json --src=$NANOCHAT_DATA_SOURCE_PATTERN --dest=$DATA_DIR
    fi
    python -m nanochat.report reset
    touch "$SETUP_FLAG"
else
    if command -v nvidia-smi &> /dev/null; then
        log "=== GPU Information Node: $(hostname) ==="
        nvidia-smi | tee -a "$LOGFILE"
    fi

    log "Waiting for Node 0 to complete setup..."
    while [ ! -f "$NANOCHAT_BASE_DIR/.setup_complete" ]; do
        sleep 5
    done
    log "Setup detected. Activating environment."
    source "$VENV_DIR/bin/activate"
fi
log "=== Starting Torchrun ==="

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=$NODE_RANK \
    -m scripts.base_train -- \
    --depth="$DEPTH" \
    --device_batch_size="$DEVICE_BATCH_SIZE" \
    --num_iterations="$NUM_ITERATIONS" \
    --eval_every="${EVAL_EVERY:-500}" \
    --core_metric_every="${CORE_METRIC_EVERY:-500}" \
    --save_every="${SAVE_EVERY:-2000}" \
    --run="$WANDB_RUN" \
    2>&1 | tee -a "$LOGFILE"

log "Training finished on Node $NODE_RANK"