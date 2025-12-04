#!/bin/env bash
#SBATCH --job-name=finewebedupl-test
#SBATCH --time=15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=1536MB
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgksdrllm-gpu-gh200

export NANOCHAT_DATA_SOURCE_PATTERN="gs://finewebedupl/runpod/full_dataset_gt_2.5/*.parquet"
export NANOCHAT_BASE_DIR="${SCRATCH}/finewebedupl"

export DEVICE_BATCH_SIZE=16
export NUM_ITERATIONS=-1
export EVAL_EVERY=250
export CORE_METRIC_EVERY=1000

mkdir -p "$NANOCHAT_BASE_DIR"

cd $NANOCHAT_BASE_DIR

if [ ! -d "nanochat-pl" ]; then
    echo "Cloning nanochat-pl..."
    git clone https://github.com/Milpo1/nanochat-pl.git
else
    echo "nanochat-pl already exists — skipping clone."
fi

cd nanochat-pl

chmod +x speedrun-pl-hpc.sh

./speedrun-pl-hpc.sh