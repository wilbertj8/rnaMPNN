#!/bin/bash
#SBATCH -t 01:30:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J training

source activate mlfold

folder_with_pdbs="/scratch/network/wj9138/new_folder"

output_dir="../outputs/example_RNA_BABY"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_output_stats=$output_dir"/output_stats.txt"

python ../helper_scripts/parse_RNA_BABY.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $output_dir \
        --num_seq_per_target 1 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1 \
        --path_to_model_weights "../training/rna_training_params/" \
        --model_name "epoch_last" \

python ../helper_scripts/model_evaluator.py --input_path=$output_dir"/seqs" --output_path=$path_for_output_stats