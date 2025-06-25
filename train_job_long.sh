#!/bin/bash

#SBATCH --job-name=cGAN-Seg-yeast-long
#SBATCH --partition=gpulong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_job_long_%j.out
#SBATCH --error=logs/train_job_long_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (adjust based on your cluster's available modules)
module purge
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/11.1.1-GCC-10.2.0

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Number of cores: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Set up temporary directory in home space to avoid /tmp issues
export TMPDIR=/home2/s5946301/tmp
mkdir -p $TMPDIR

# Navigate to the project directory
cd /home2/s5946301/cGAN-Seg-yeast

# Activate virtual environment if you have one
# source venv/bin/activate  # Uncomment if you use a virtual environment

# Print Python and CUDA information
echo "Python version:"
python --version
echo "CUDA devices:"
nvidia-smi
echo ""

# Run the training command with optimized patience
echo "Starting cGAN-Seg training (with early stopping)..."
python train.py \
    --seg_model DeepSea \
    --train_set_dir yeaz_dataset/train/ \
    --lr 0.0001 \
    --p_vanilla 0.2 \
    --p_diff 0.2 \
    --patience 200 \
    --output_dir tmp/

echo ""
echo "Job finished at: $(date)"
