#!/bin/bash
#SBATCH --job-name=rna_test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

source activate mlfold

folder_with_pdbs="../inputs/RNA_BABY"

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
        --sampling_temp "0.1 0.2" \
        --seed 37 \
        --batch_size 1 \
        --model_name "v_RNA"

python ../helper_scripts/model_evaluator.py --input_path=$output_dir"/seqs" --output_path=$path_for_output_stats