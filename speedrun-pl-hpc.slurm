#!/bin/bash -l
#SBATCH --job-name=finewebedupl-test
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=1536MB
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgksdrllm-gpu-gh200

module purge

module load CUDA/12.8.0 ML-bundle/25.04

module list

export NANOCHAT_DATA_SOURCE_PATTERN="gs://finewebedupl/runpod/full_dataset_gt_2.5/*.parquet"
export FINEWEB_BASE_DIR="${SCRATCH}/finewebedupl2"
export NANOCHAT_BASE_DIR="$/workspace/nanochat-pl"

export DEVICE_BATCH_SIZE=32
export NUM_ITERATIONS=40000
export EVAL_EVERY=500
export CORE_METRIC_EVERY=500
export SAVE_EVERY=2000
export WANDB_API_KEY=""

mkdir -p "$FINEWEB_BASE_DIR"

cd $FINEWEB_BASE_DIR

if [ ! -d "nanochat-pl" ]; then
    echo "Cloning nanochat-pl..."
    git clone https://github.com/Milpo1/nanochat-pl.git
else
    echo "nanochat-pl already exists — skipping clone."
fi

cd nanochat-pl

chmod +x speedrun-pl-hpc.sh

./speedrun-pl-hpc.sh