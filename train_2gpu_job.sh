#!/bin/bash

#SBATCH --job-name=cGAN-Seg-2gpu-v100
#SBATCH --partition=gpulong
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/s5946301/cGAN-Seg-yeast/logs/train_2gpu_%j.out
#SBATCH --error=/home2/s5946301/cGAN-Seg-yeast/logs/train_2gpu_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Number of cores: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Number of GPUs: $SLURM_GPUS_PER_NODE"
echo "Start time: $(date)"
echo ""

# Set up temporary directory in home space
export TMPDIR=/home2/s5946301/tmp
mkdir -p $TMPDIR

# Navigate to the project directory
cd /home2/s5946301/cGAN-Seg-yeast

# Activate virtual environment
source venv/bin/activate

# Print Python and CUDA information
echo "Python version:"
python --version
echo "CUDA devices:"
nvidia-smi
echo ""

# Multi-GPU training environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Create output directory
mkdir -p tmp_2gpu

# Run 2-GPU training using torchrun
echo "Starting 2-GPU cGAN-Seg training (V100)..."
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_distributed.py \
    --seg_model DeepSea \
    --train_set_dir yeaz_dataset/train/ \
    --lr 0.0002 \
    --p_vanilla 0.2 \
    --p_diff 0.2 \
    --patience 500 \
    --batch_size 6 \
    --output_dir tmp_2gpu/

echo ""
echo "2-GPU training finished at: $(date)"
