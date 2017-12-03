import os
import argparse
import numpy as np
from skbio.tree import TreeNode
from io import StringIO
from skbio import DistanceMatrix
from skbio.tree import nj
import shutil
import pandas as pd

def run_nj_get_dist_matrix(dist_matrix):
    dm = DistanceMatrix(dist_matrix)

    # run neighbor join and get dist matrix from the tree
    nj_tree = nj(dm)
    df = nj_tree.tip_tip_distances().to_data_frame()
    df.index = df.index.astype(int)

    # sort rows and cols
    df.sort_index(inplace=True)
    df.columns = df.columns.values.astype(np.int32)
    df = df[sorted(df.columns)]

    return df.as_matrix()

def get_dist_matrix_from_tree(file_path):
    with open(file_path, 'r') as myfile:
        data = myfile.read().replace('\n', '')
        t = TreeNode.read(StringIO(data))
        df = t.tip_tip_distances().to_data_frame()

        #df.index = df.index.astype(int)

        # sort rows and cols
        df.sort_index(inplace=True)
        #df.columns = df.columns.values.astype(np.int32)
        df = df[sorted(df.columns)]
        print(df)
        return df.as_matrix()

def calculate_fitness(matrix1, matrix2):
    return np.square(matrix1 - matrix2).sum()

def convert_to_phylip_format(args):
    directory_path = args.input
    directory = os.fsencode(directory_path)

    for file in os.listdir(directory):
        object_index = 0
        filename = os.fsdecode(file)
        if filename.endswith("additive.noisy.txt"):
            file_path = os.path.join(directory_path, filename)
            dim_matrix = sum(1 for line in open(file_path))
            file_lines = list()
            file_lines.append('     ' + str(dim_matrix) + '\n')
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    object_name = 'object' + "{0:03}".format(object_index) + ' '
                    object_index += 1
                    file_lines.append(object_name + line)

            with open(file_path.replace("additive.noisy.txt", "additive.noisy.phylip.txt"), 'w') as f:
                f.writelines(file_lines)


def run_phylip(args):
    results = pd.DataFrame(columns=['matrix', 'location', 'nj_fitness', 'fitch_fitness'])

    directory_path = args.input
    directory = os.fsencode(directory_path)
    i = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("additive.noisy.phylip.txt"):
            file_path = os.path.join(directory_path, filename)
            input_matrix = np.load(file_path.replace("additive.noisy.phylip.txt", 'additive.noisy.npy'))
            print('Processing:', file_path)

            # copy to infile
            shutil.copy(file_path, 'infile')

            # run fitch
            os.system('./phylip_linux/phylip-3.696/exe/fitch < input_fitch')

            # copy out outputfile / outputtree
            fitch_dist_matrix = get_dist_matrix_from_tree('outtree')
            nj_dist_matrix = run_nj_get_dist_matrix(input_matrix)

            fitch_fitness = calculate_fitness(fitch_dist_matrix, input_matrix)
            nj_fitness = calculate_fitness(nj_dist_matrix, input_matrix)
            print("fitness fitch:", fitch_fitness)
            print("fitness nj:", nj_fitness)
            results.loc[i] = [filename.replace("additive.noisy.phylip.txt",""), file_path, nj_fitness, fitch_fitness]
            i += 1

            shutil.copy('outfile', file_path.replace("additive.noisy.phylip.txt", "additive.noisy.fitch.outfile"))
            shutil.copy('outtree', file_path.replace("additive.noisy.phylip.txt", "additive.noisy.fitch.outtree"))

            # remove output and input files
            os.remove('infile')
            os.remove('outtree')
            os.remove('outfile')

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert numpy to phylip data input format')
    parser.add_argument('--input', help='input dir')
    args = parser.parse_args()

    # generate phylip format data
    convert_to_phylip_format(args)

    # run experiments
    results = run_phylip(args)

    # Header is: 'matrix', 'location', 'nj_fitness', 'fitch_fitness'
    with open('results_phylip.csv', 'a') as f:
        results.to_csv(f, header=False)

    print('Results outputted to results_phylip.csv')
