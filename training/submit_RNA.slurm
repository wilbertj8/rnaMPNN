#!/bin/bash
#SBATCH -t 03:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J training

source activate mlfold
python ./rna_training.py \
           --path_for_outputs "./rna_weights/" \
           --path_for_training_data "./dataset/" \
           --previous_checkpoint "./rna_training_params/v_RNA_0.pt" \
           --num_examples_per_epoch 64 \
           --save_model_every_n_epochs 50 \
           --num_epochs 400 \
          