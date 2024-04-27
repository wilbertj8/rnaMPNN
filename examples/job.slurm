#!/bin/bash
#SBATCH -t 03:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J training

source activate mlfold

folder_with_pdbs="../inputs/RNA_stuff"

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
        --num_seq_per_target 2 \
        --sampling_temp "0.1 0.15 0.2 0.25 0.3" \
        --seed 37 \
        --batch_size 1 \
        --path_to_model_weights "../training/rna_weights/model_weights/" \
        --model_name "epoch_last" \

python ../helper_scripts/model_evaluator.py --input_path=$output_dir"/seqs" --output_path=$path_for_output_stats