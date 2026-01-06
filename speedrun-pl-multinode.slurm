#!/bin/bash -l
#SBATCH --job-name=finewebedupl-d32-run-3-multinode-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=1536MB
#SBATCH --time=1:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgksdrllm-gpu-gh200

module purge
module load CUDA/12.8.0 ML-bundle/25.04

# 1. Get the list of nodes allocated
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

# 2. Get the IP address of the head node
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | grep -oP '172\.25\.[0-9]+\.[0-9]+' | head -n 1)
export MASTER_PORT=29500
export RDZV_ID=$SLURM_JOB_ID

echo "Master Node: $head_node ($MASTER_ADDR:$MASTER_PORT)"
echo "Node List: $SLURM_JOB_NODELIST"

export FINEWEB_BASE_DIR="${SCRATCH}/run-3-multinode-test"
export NANOCHAT_BASE_DIR="${FINEWEB_BASE_DIR}/nanochat-pl"
export NANOCHAT_DATA_SOURCE_PATTERN="gs://finewebedupl/training_dataset/fineweb2_finepdfs_filtered_ge25_rehydrated_score_gamma075_38B_shuffled/000*.parquet"

export DEVICE_BATCH_SIZE=8
export NUM_ITERATIONS=68500
export EVAL_EVERY=1000
export CORE_METRIC_EVERY=2000
export SAVE_EVERY=5000

mkdir -p "$FINEWEB_BASE_DIR"

cd $FINEWEB_BASE_DIR

if [ ! -d "nanochat-pl" ]; then
    echo "Cloning nanochat-pl..."
    git clone https://github.com/Milpo1/nanochat-pl.git
fi

mkdir -p "${NANOCHAT_BASE_DIR}/logs"
cd $NANOCHAT_BASE_DIR

SCRIPT_PATH="${NANOCHAT_BASE_DIR}/speedrun-pl-multinode.sh"

chmod +x $SCRIPT_PATH

srun bash $SCRIPT_PATH