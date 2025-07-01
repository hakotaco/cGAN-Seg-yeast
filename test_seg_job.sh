#!/bin/bash

#SBATCH --job-name=cGAN-Seg-test-seg
#SBATCH --partition=gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home2/s5946301/cGAN-Seg-yeast/logs/test_seg_%j.out
#SBATCH --error=/home2/s5946301/cGAN-Seg-yeast/logs/test_seg_%j.err

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
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
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

# Create output directory for test results
mkdir -p test_results_seg

# Run the segmentation model testing
echo "Starting segmentation model testing..."
python test_segmentation_model.py \
    --seg_model DeepSea \
    --test_set_dir yeaz_dataset/test/ \
    --seg_ckpt_dir tmp_long/Seg.pth \
    --output_dir test_results_seg/

echo ""
echo "Segmentation testing finished at: $(date)"
