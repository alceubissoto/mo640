import numpy as np
from skbio import DistanceMatrix
from skbio.tree import nj
import argparse
import os
import bottleneck as bn
import time
from tqdm import tqdm
import pandas as pd

NUM_PARENTS = 5
NUM_CHILDREN = 2
NUM_ITERATIONS = 500
SIGMA = 5

class Individual:
    dist_matrix = None
    fitness_score = 0 # sum of the squares of distances between this individual and ground truth matrix

    def __init__(self, dist_matrix, fitness):
        self.dist_matrix = dist_matrix
        self.fitness_score = fitness


class GeneticAlgorithmRunner:
    ground_truth_matrix = None # additive matrix with gaussian noise
    seed_matrix = None # dist matrix returned after first
    num_children = 0 # number of children each parent will breed
    num_iterations = 0 # number of iterations to run before returning
    num_parents = 0 # number of descendants to carry from each iteration to the next
    population = list()

    def __init__(self,
                 ground_truth_matrix,
                 num_parents,
                 num_children,
                 num_iterations):
        self.ground_truth_matrix = ground_truth_matrix
        self.num_iterations = num_iterations
        self.num_children = num_children
        self.num_parents = num_parents

    def run_nj_get_dist_matrix(self, dist_matrix):
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

    def get_seed_matrix(self):
        '''
        Run neighbor join once and return dist matrix
        :return:
        '''
        if self.seed_matrix is None:
            self.seed_matrix = self.run_nj_get_dist_matrix(self.ground_truth_matrix)

        return self.seed_matrix

    def mutate(self, dist_matrix):
        '''
        Returns an object of Individual type
        :return:
        '''
        raise NotImplementedError("Please Implement this method in child class")

    def breed(self):
        '''
        Returns list of the entire population = parents + children
        :return:
        '''
        raise NotImplementedError("Please Implement this method in child class")

    def select(self):
        '''
        Select and maintain in the population only the best individuals
        '''

        self.population.sort(key=lambda x: x.fitness_score)

        # print('select before:', len(self.population))
        # print('before selection:', [x.fitness_score for x in self.population])
        del self.population [self.num_parents:]
        # print('after selection:', [x.fitness_score for x in self.population])
        # print('select after:', len(self.population))

    def calculate_fitness(self, mutated_matrix):
        dist_matrix = self.run_nj_get_dist_matrix(mutated_matrix)
        return np.square(self.ground_truth_matrix - dist_matrix).sum()

    def add_to_population(self, dist_matrix):
        self.population.append(Individual(dist_matrix, self.calculate_fitness(dist_matrix)))

    def print_population(self):
        print("------------ Population ---------------")
        for i in self.population:
            print(i.dist_matrix)

    def start_population(self):
        seed_matrix = self.get_seed_matrix()
        self.add_to_population(seed_matrix)

        while len(self.population) < self.num_parents:
            seed_descendant = self.mutate(seed_matrix)
            self.add_to_population(seed_descendant)

        #self.print_population()

    def get_best_fitness(self):
        return min(i.fitness_score for i in self.population)

    def run(self):
        # results in a numpy matrix
        results = pd.DataFrame(columns=['iteration', 'timestamp', 'best_fitness'])

        # breed a minimum number of parents to get started
        self.start_population()

        # iterate between breed and select
        for i in tqdm(range(self.num_iterations)):
            self.breed()
            self.select()
            results.loc[i] = [i, time.time(), self.get_best_fitness()]

        return results


class OptimizeMatrixCellGeneticRunner(GeneticAlgorithmRunner):
    num_cells_to_mutate = 1

    def top_n_indexes(self, arr, n):
        idx = bn.argpartsort(arr, arr.size - n, axis=None)[-n:]
        width = arr.shape[1]
        return [divmod(i, width) for i in idx]

    ''''
    Algorithm #3: only mutate the cells that are more distant from
    the ground truth matrix
    '''
    def breed(self):
        # print("start breed")
        # create new children
        new_children = list()
        for individual in self.population:
            for i in range(self.num_children):
                child = self.mutate(individual.dist_matrix)
                new_children.append(child)

        # add to population
        for child in new_children:
            self.add_to_population(child)

    def mutate(self, dist_matrix):
        diff_matrix = np.square(self.ground_truth_matrix - dist_matrix)
        child_matrix = np.copy(dist_matrix)
        noise = np.random.normal(loc=0.0, scale=SIGMA)

        # this returns a list of indices of the highest elements
        # the indices are the matrix in a flattened array
        # we are multiplying by 2 because this is a symmetric matrix so there are 2
        # equal values always
        indices = np.argsort(diff_matrix, axis=None)[-(self.num_cells_to_mutate*2):]
        for ind in indices:
            i, j = np.unravel_index(ind, diff_matrix.shape)

            # add noise to the cell with highest distance from the ground truth
            child_matrix[i, j] = child_matrix[i, j] + noise

        return child_matrix


def main(args):
    directory_path = args.in_data
    directory = os.fsencode(directory_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("additive.noisy.npy"):
            dataset_matrix = np.load(os.path.join(directory_path, filename))
            runner = OptimizeMatrixCellGeneticRunner(ground_truth_matrix=dataset_matrix,
                                                     num_iterations=NUM_ITERATIONS,
                                                     num_children=NUM_CHILDREN,
                                                     num_parents=NUM_PARENTS
                                                     )
            results = runner.run()
            results['dataset'] = directory_path
            results['matrix'] = filename
            with open('run_results.csv', 'a') as f:
                results.to_csv(f, header=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic algorithms to create phylogenetic trees.')
    parser.add_argument('--in-data', help='runs genetic algorithms using source data')
    args = parser.parse_args()
    main(args)

