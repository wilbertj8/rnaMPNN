#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 2
#SBATCH --output=parse_multiple_chains.out

source activate mlfold

folder_with_pdbs="/Users/wilbertjoseph/Downloads/IW/solo_representative_pdb_all__3_326/"

python create_pt_files.py --input_path=$folder_with_pdbs --output_path='../dataset/pdb/'
