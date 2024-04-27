#!/bin/bash
#SBATCH --job-name=rna_test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)

source activate mlfold
python ./rna_training.py \
           --path_for_outputs "./rna_weights/" \
           --path_for_training_data "./dataset/" \
           --previous_checkpoint "./rna_weights/model_weights/epoch_last.pt" \
           --num_examples_per_epoch 64 \
           --save_model_every_n_epochs 50
