#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=128g
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH -t 7-00:00:00
#SBATCH --output=exp_020.out

source activate mlfold
python ./rna_training.py \
           --path_for_outputs "./rna_weights/" \
           --path_for_training_data "./dataset/" \
           --previous_checkpoint "./rna_weights/model_weights/epoch_last.pt" \
           --num_examples_per_epoch 64 \
           --save_model_every_n_epochs 50
