#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J training

echo "Running on host: $(hostname)"
echo "We have allocated $(nproc) CPU core(s)."
echo "Starting the CUDA test script."

module load cuda/11.3  # Adjust based on available versions

python slurm_test.py
