#!/bin/bash
#SBATCH -A f202509580cpcaa1g
#SBATCH -t 24:00:00
#SBATCH -p normal-a100-80
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=/projects/F202509580CPCAA1/pedrofsousa/meddiff/logs/run_V0_deucalion/output_V0_deucalion.out
#SBATCH --error=/projects/F202509580CPCAA1/pedrofsousa/meddiff/logs/run_V0_deucalion/error_V0_deucalion.err

# CMD Script
#cd /projects/F202509580CPCAA1/pedrofsousa/meddiff/runs         # ALWAYS
#module load Anaconda3/2025.06-1                                # IT NOT IN BASE ENV
#conda create --name meddiff python=3.8                         # ONCE
#conda env create --name meddiff --file=/projects/F202509580CPCAA1/pedrofsousa/meddiff/conda_requirements.yml
#source /eb/x86_64/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
#conda init --all
#conda activate meddiff
#conda install --file /projects/F202509580CPCAA1/pedrofsousa/meddiff/conda_requirements.yml
#pip install -r /projects/F202509580CPCAA1/pedrofsousa/meddiff/pip_requirements.txt

# -------------------------------------------------------------

# Module Loading
module load nvidia-compilers/25.1
module load CUDA/11.8.0
module load Anaconda3/2025.06-1
source /eb/x86_64/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
conda activate meddiff

# WandB Configuration
export WANDB_MODE="offline"
export WANDB_DATA_DIR="/projects/F202509580CPCAA1/pedrofsousa/meddiff/logs/wandb"
mkdir -p /projects/F202509580CPCAA1/pedrofsousa/meddiff/logs/wandb

# Distributed Training Configuration
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
readarray -t NODES_ARRAY <<< "$NODES"
HEAD_NODE=${NODES_ARRAY[0]}
export MASTER_ADDR=$HEAD_NODE
get_free_port() {
  while true; do
    PORT=$((10000 + RANDOM % 50000))
    ss -ltn | awk '{print $4}' | grep -q ":$PORT" || break
  done
  echo $PORT
}
export MASTER_PORT=$(get_free_port)

# GPU Configuration
NUM_GPUS=$(nvidia-smi -L | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

echo "GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Distributed Run Execution
python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /projects/F202509580CPCAA1/pedrofsousa/meddiff/runs/main_V0_deucalion.py
