#!/bin/bash
#SBATCH -t 00:05:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J training

source activate mlfold

folder_with_pdbs="../../solo_representative_pdb_all__3_326/"

python create_pt_files.py --input_path=$folder_with_pdbs --output_path='../dataset/pdb/'
