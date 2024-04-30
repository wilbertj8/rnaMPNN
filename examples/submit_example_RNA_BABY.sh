#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 2
#SBATCH --output=example_1.out

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
        --num_seq_per_target 1 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1 \
        --model_name "v_RNA"

python ../helper_scripts/model_evaluator.py --input_path=$output_dir"/seqs" --output_path=$path_for_output_stats