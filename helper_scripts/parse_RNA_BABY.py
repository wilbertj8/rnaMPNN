import argparse

def main(args):
    import numpy as np
    import os, time, gzip, json
    import glob 
    
    folder_with_pdbs_path = args.input_path
    save_path = args.output_path
    ca_only = args.ca_only
    
    alpha_1 = list("ACGUX-")
    states = len(alpha_1)
    alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
               'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
    
    aa_1_N = {a:n for n,a in enumerate(alpha_1)}
    aa_3_N = {a:n for n,a in enumerate(alpha_3)}
    aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
    aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
    aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
    
    def AA_to_N(x):
        # ["ARND"] -> [[0,1,2,3]]
        x = np.array(x)
        if x.ndim == 0: x = x[None]
        return [[aa_1_N.get(a, states-1) for a in y] for y in x]
    
    def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x)
        if x.ndim == 1: x = x[None]
        return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]
    
    
    def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
        '''
        input:  x = PDB filename
                atoms = atoms to extract (optional)
        output: (length, atoms, coords=(x,y,z)), sequence
        '''
        xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
        for line in open(x,"rb"):
            line = line.decode("utf-8","ignore").rstrip()

            if line[:6] == "HETATM" and line[17:17+3] != "HOH":
                line = line.replace("HETATM","ATOM  ")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None:
                    atom = line[12:12+4].strip()
                    nucleotide = line[17:17+3].strip()
                    nucleotide = "X" if len(nucleotide) > 1 else nucleotide
                    nuc_num = line[22:22+5].strip()
                    x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

                    if nuc_num[-1].isalpha(): 
                        nuc_a,nuc_num = nuc_num[-1],int(nuc_num[:-1])-1
                    else: 
                        nuc_a,nuc_num = "",int(nuc_num)-1
                    # nuc_num = int(nuc_num)
                    if nuc_num < min_resn: 
                        min_resn = nuc_num
                    if nuc_num > max_resn: 
                        max_resn = nuc_num
                    
                    if nuc_num not in xyz: 
                        xyz[nuc_num] = {}
                    if nuc_a not in xyz[nuc_num]: 
                        xyz[nuc_num][nuc_a] = {}
                    if atom not in xyz[nuc_num][nuc_a]:
                        xyz[nuc_num][nuc_a][atom] = np.array([x,y,z])

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
                if nuc_num in xyz:
                    for k in sorted(xyz[nuc_num]):
                        for atom in atoms:
                            if atom in xyz[nuc_num][k]:
                                xyz_.append(xyz[nuc_num][k][atom])
                            else:
                                xyz_.append(np.full(3,np.nan))
                else:
                    for atom in atoms:
                        xyz_.append(np.full(3,np.nan))
            return np.array(xyz_).reshape(-1,len(atoms),3), ["".join(seq_)]
        
        except TypeError:
            return 'no_chain', 'no_chain'
    
    
    
    pdb_dict_list = []
    c = 0
    
    if folder_with_pdbs_path[-1]!='/':
        folder_with_pdbs_path = folder_with_pdbs_path+'/'
    
    
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    
    biounit_names = glob.glob(folder_with_pdbs_path+'*.pdb')
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            atoms = ['P', 'O5\'', 'O3\'', 'C5\'', 'C4\'', 'C3\'', 'C2\'', 'C1\'']
            xyz, seq = parse_PDB_biounits(biounit, atoms=atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_'+letter]=seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
                else:
                    for i, atom in enumerate(atoms):
                        coords_dict_chain[f'{atom}_chain_{letter}'] = xyz[:, i, :].tolist()
                my_dict['coords_chain_'+letter]=coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name']=biounit[(fi+1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s < len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c+=1
            
    with open(save_path, 'w') as f:
       for entry in pdb_dict_list:
           f.write(json.dumps(entry) + '\n')
           

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--input_path", type=str, help="Path to a folder with pdb files, e.g. /home/my_pdbs/")
    argparser.add_argument("--output_path", type=str, help="Path where to save .jsonl dictionary of parsed pdbs")
    argparser.add_argument("--ca_only", action="store_true", default=False, help="parse a backbone-only structure (default: false)")

    args = argparser.parse_args()
    main(args)
