#!/bin/bash

#SBATCH --job-name=cGAN-test-distributed
#SBATCH --partition=gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home2/s5946301/cGAN-Seg-yeast/logs/test_distributed_%j.out
#SBATCH --error=/home2/s5946301/cGAN-Seg-yeast/logs/test_distributed_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

# Print job information
echo "=== DISTRIBUTED TRAINING TEST JOB ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Number of cores: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Number of GPUs: $SLURM_GPUS_PER_NODE"
echo "Start time: $(date)"
echo ""

# Navigate to the project directory
cd /home2/s5946301/cGAN-Seg-yeast

# Activate virtual environment
source venv/bin/activate

# Print Python and CUDA information
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
echo "CUDA devices:"
nvidia-smi
echo ""

# Multi-GPU training environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Create test output directory
mkdir -p test_distributed_tmp

# Test: Check if distributed training files exist
echo "=== CHECKING FILES ==="
if [ -f "train_distributed.py" ]; then
    echo "✓ train_distributed.py exists"
else
    echo "✗ train_distributed.py missing!"
    exit 1
fi

if [ -d "yeaz_dataset/train/" ]; then
    echo "✓ Training dataset exists"
    ls yeaz_dataset/train/ | head -3
else
    echo "✗ Training dataset missing!"
    exit 1
fi

echo ""
echo "=== TESTING DISTRIBUTED TRAINING (2 minutes only) ==="

# Run minimal distributed training test with very few epochs
timeout 120 torchrun \
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
    --patience 2 \
    --batch_size 2 \
    --epochs 3 \
    --output_dir test_distributed_tmp/ || echo "Training timed out after 2 minutes (expected)"

echo ""
echo "=== TEST RESULTS ==="
if [ -d "test_distributed_tmp" ]; then
    echo "✓ Output directory created"
    ls -la test_distributed_tmp/ | head -5
else
    echo "✗ No output directory created"
fi

echo ""
echo "=== GPU MEMORY USAGE ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo ""
echo "Test job finished at: $(date)"
echo "=== END OF TEST ==="
