#!/bin/bash
#SBATCH --job-name=rna_test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)

echo "Running on host: $(hostname)"
echo "We have allocated $(nproc) CPU core(s)."
echo "Starting the CUDA test script."

module load python/3.12.3  # Adjust as necessary to load Python with PyTorch installed
module load cuda/12.3   # Adjust the version based on what's available on your system

python test_cuda.py
