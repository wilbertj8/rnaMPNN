import argparse, time

def main(args):
    import numpy as np
    import glob
    import torch
    import csv
    
    folder_with_pdbs_path = args.input_path
    
    def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
        '''
        input:  x = PDB filename
                atoms = atoms to extract (optional)
        output: (length, atoms, coords=(x,y,z)), sequence
        '''
        seq,min_resn,max_resn = {},1e6,-1e6
        for line in open(x,"rb"):
            line = line.decode("utf-8","ignore").rstrip()

            if line[:6] == "HETATM" and line[17:17+3] != "HOH":
                line = line.replace("HETATM","ATOM  ")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None:
                    nucleotide = line[17:17+3].strip()
                    nucleotide = "X" if len(nucleotide) > 1 else nucleotide
                    nuc_num = line[22:22+5].strip()

                    if nuc_num[-1].isalpha(): 
                        nuc_a,nuc_num = nuc_num[-1],int(nuc_num[:-1])-1
                    else: 
                        nuc_a,nuc_num = "",int(nuc_num)-1
                    # nuc_num = int(nuc_num)
                    if nuc_num < min_resn: 
                        min_resn = nuc_num
                    if nuc_num > max_resn: 
                        max_resn = nuc_num

                    if nuc_num not in seq: 
                        seq[nuc_num] = {}
                    if nuc_a not in seq[nuc_num]: 
                        seq[nuc_num][nuc_a] = nucleotide
    
        # convert to numpy arrays, fill in missing values
        seq_,xyz_ = [],[]
        try:
            for nuc_num in range(min_resn,max_resn+1):
                if nuc_num in seq:
                    for k in sorted(seq[nuc_num]):
                        seq_.append(seq[nuc_num][k])
                else:
                    seq_.append('X')
            return ["".join(seq_)]
        
        except TypeError:
            return 'no_chain'
    
    if folder_with_pdbs_path[-1]!='/':
        folder_with_pdbs_path = folder_with_pdbs_path+'/'
    
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
                     'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j',
                     'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    atoms = ['P', 'O5\'', 'O3\'', 'C5\'', 'C4\'', 'C3\'', 'C2\'', 'C1\'']
    

    # create a csv file with the following column titles: CHAINID, CLUSTER
    csv_file = args.output_path + 'list.csv'
    column_titles = ['CHAINID', 'CLUSTER', 'SEQUENCE']

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_titles)

    cluster = 0
    biounit_names = glob.glob(folder_with_pdbs_path+'*.pdb')
    for biounit in biounit_names:
        name = biounit.split('/')[-1].split('.')[0]
        name = name.split('_')[0]
        print(name)
        for letter in chain_alphabet:
            seq = parse_PDB_biounits(biounit, atoms=atoms, chain=letter)
            if type(seq) != str:
                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f'{name}_{letter}', cluster, seq[0]])
                cluster += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--input_path", type=str, help="Path to a folder with pdb files, e.g. /home/my_pdbs/")
    argparser.add_argument("--output_path", type=str, help="Path where to save .jsonl dictionary of parsed pdbs")

    args = argparser.parse_args()
    
    start_time = time.time()
    
    main(args)
    
    end_time = time.time()

    print(f"The function took {end_time - start_time} seconds to complete.")
