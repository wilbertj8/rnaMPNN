#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J training

echo "Running on host: $(hostname)"
echo "We have allocated $(nproc) CPU core(s)."
echo "Starting the CUDA test script."

module load cuda/12.3  # Adjust based on available versions
module load python/3.12.3  # Ensure it matches your Conda environment's Python version

python slurm_test.py
