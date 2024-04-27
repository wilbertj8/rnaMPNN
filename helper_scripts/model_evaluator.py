'''
Script to evaluate the performance of rnaMPNN based on output files
'''

import argparse

def main(args):

    import glob
    import numpy as np

    recovery_rates = {}

    # Get all files in the input folder
    biounit_names = glob.glob(args.input_path + "/*.fa")

    for biounit in biounit_names:
        rna = biounit.split('/')[-1].split('.')[0]
        recovery_rates[rna] = {}

        with open(biounit, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('>'):
                    if line[1] == "T":
                        stats = line.split(",")
                        temp = float(stats[0].split("=")[1])
                        seq_recovery = float(stats[-1].split("=")[1].strip())
                        if temp not in recovery_rates[rna]:
                            recovery_rates[rna][temp] = []
                        recovery_rates[rna][temp].append(seq_recovery)
                elif "size" not in recovery_rates[rna]:
                    recovery_rates[rna]["size"] = len(line.strip())

    # output stats to text file
    with open(args.output_path, 'w') as f:
        total_sum = 0
        total_count = 0
        for rna in recovery_rates:
            if not recovery_rates[rna]:
                continue

            rna_sum = 0
            rna_count = 0

            f.write(f"RNA: {rna} (Size: {recovery_rates[rna]["size"]}) \n")
            f.write('\n')

            for temp in recovery_rates[rna]:
                if temp == "size":
                    continue

                temp_sum = np.sum(recovery_rates[rna][temp])
                temp_count = len(recovery_rates[rna][temp])
                temp_mean = temp_sum / temp_count

                rna_sum += temp_sum
                rna_count += temp_count

                out = np.format_float_positional(temp_mean, unique=False, precision=4)

                f.write("Temp: " + str(temp) + '\n')
                f.write("Mean: " + str(out) + '\n')
            
            rna_mean = rna_sum / rna_count

            out = np.format_float_positional(rna_mean, unique=False, precision=4)
            f.write('\n')
            f.write("AVERAGE: " + str(out) + '\n')
            f.write('\n')

            total_sum += rna_sum
            total_count += rna_count
        
        total_mean = total_sum / total_count

        out = np.format_float_positional(total_mean, unique=False, precision=4)
        f.write("OVERALL AVERAGE: " + str(out) + '\n')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--input_path", type=str, help="Path to a folder with sequence files, e.g. /home/outputs/")
    argparser.add_argument("--output_path", type=str, help="Path where to save text file with results")

    args = argparser.parse_args()

    main(args)