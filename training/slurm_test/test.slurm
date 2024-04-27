#!/bin/bash
#SBATCH -t 00:05:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J training

echo "Running on host: $(hostname)"
echo "We have allocated $(nproc) CPU core(s)."
echo "Starting the CUDA test script."

printenv > env_output.txt

python slurm_test.py
