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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=INFO

# Log file specific to this node
LOG_DIR="$NANOCHAT_BASE_DIR/logs"
LOGFILE="$LOG_DIR/run_${SLURM_JOB_ID}_node_${NODE_RANK}.log"

log() {
    echo "[Node $NODE_RANK] [$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "Starting on $(hostname)"
log "Role: Node $NODE_RANK of $NNODES"
log "Master: $MASTER_ADDR:$MASTER_PORT"

VENV_DIR="$NANOCHAT_BASE_DIR/.venv"
SETUP_FLAG="$NANOCHAT_BASE_DIR/.setup_complete"

if [ "$NODE_RANK" -eq 0 ]; then
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
    uv sync --extra gpu
    
    # Only Node 0 downloads
    DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
        log "Downloading data..."
        uv pip install google-cloud-storage
        python gcp_fetch.py --key=$HOME/.keys/gcs-read-only.json --src=$NANOCHAT_DATA_SOURCE_PATTERN --dest=$DATA_DIR
    fi
    
    touch "$SETUP_FLAG"
else
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