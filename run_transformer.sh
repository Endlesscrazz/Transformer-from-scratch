#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=trans-en-hi-gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1527145@utah.edu  

# 1) cd into your scratch directory
cd /scratch/general/vast/u1527145/Transformer-from-scratch

# 2) Load Miniconda module for a clean environment
module purge
module load miniconda3/23.11.0

# 3) Prepend projenv's bin directory to the PATH to activate the environment
export PATH="$HOME/.conda/envs/projenv/bin:${PATH}"

# 4) Verify that we’re using the correct Python and that Torch can see the GPU
echo "--- Verifying Environment ---"
which python
python - <<<'import torch; print("Torch version:", torch.__version__, "| CUDA available:", torch.cuda.is_available())'
echo "---------------------------"

# 5) Redirect HF caches into scratch to avoid home‐dir quotas
export HF_DATASETS_CACHE="/scratch/general/vast/u1527145/Transformer-from-scratch/hf_cache/datasets"
export HF_METRICS_CACHE="/scratch/general/vast/u1527145/Transformer-from-scratch/metrics"
export TOKENIZERS_PARALLELISM=false

# 6) Ensure logs/ and weights/ directories exist
mkdir -p logs
mkdir -p weights

# 7) Print GPU debug info
echo "--- Job Info ---"
echo "Running on host $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "----------------"

# 8) Run your training script
# The arguments here will now correctly override the defaults in config.py
echo "--- Starting Training ---"
python train.py \
    --batch-size 16 \
    --num-epochs 20 \
    --learning-rate 1e-4

echo "=== Training finished on $(date) ==="