import argparse

def main(args):
    import glob
    import random
    import numpy as np
    import json
    
    mpnn_alphabet = 'ACGUX'
    
    mpnn_alphabet_dict = {'A': 0,'C': 1,'G': 2,'U': 3,'X': 4}
     
    with open(args.input_path, 'r') as json_file:
        json_list = list(json_file)   
 
    my_dict = {}
    for json_str in json_list:
        result = json.loads(json_str)
        all_chain_list = [item[-1:] for item in list(result) if item[:10]=='seq_chain_']
        bias_by_res_dict = {}
        for chain in all_chain_list:
            chain_length = len(result[f'seq_chain_{chain}'])
            bias_per_residue = np.zeros([chain_length, 5])


            if chain == 'A':
                residues = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
                amino_acids = [5, 9] #[G, L]
                for res in residues:
                    for aa in amino_acids:
                        bias_per_residue[res, aa] = 100.5

            if chain == 'C':
                residues = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
                amino_acids = range(21)[1:] #[G, L]
                for res in residues:
                    for aa in amino_acids:
                        bias_per_residue[res, aa] = -100.5

            bias_by_res_dict[chain] = bias_per_residue.tolist()
        my_dict[result['name']] = bias_by_res_dict

    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
    argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")

    args = argparser.parse_args()
    main(args) 
