import numpy as np
from skbio import DistanceMatrix
from skbio.tree import nj
import argparse
import os
import time
from tqdm import tqdm
import pandas as pd
import itertools
import random

NUM_PARENTS = 5
NUM_CHILDREN = 2
#NUM_ITERATIONS = 500
SIGMA = 5
NUM_CELLS_TO_MUTATE = 1


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
        self.population = list()
        self.seed_matrix = None

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
        already_mutated = set()

        diff_matrix = np.square(self.ground_truth_matrix - dist_matrix)
        child_matrix = np.copy(dist_matrix)

        # this returns a list of indices of the highest elements
        # the indices are the matrix in a flattened array
        # we are multiplying by 2 because this is a symmetric matrix so there are 2
        # equal values always
        # diff_matrix is negative here cause I want argsort in reverse order
        indices = np.argsort(-diff_matrix, axis=None)
        for ind in indices:
            if len(already_mutated) > NUM_CELLS_TO_MUTATE*2:
                # we have already mutated enough cells, stop
                break

            i, j = np.unravel_index(ind, diff_matrix.shape)
            if (i, j) not in already_mutated:
                noise = np.random.normal(loc=0.0, scale=SIGMA)

                # add noise to the cell with highest distance from the ground truth
                child_matrix[i, j] += noise
                child_matrix[j, i] += noise
                already_mutated.add((i, j))
                already_mutated.add((j, i))

        return child_matrix

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
        del self.population [self.num_parents:]

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

        # breed a sufficient number of children to get started
        while len(self.population) < self.num_parents:
            seed_descendant = self.mutate(seed_matrix)
            self.add_to_population(seed_descendant)

    def get_best_fitness_from_population(self):
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
            results.loc[i] = [i, time.time(), self.get_best_fitness_from_population()]

        return results


class OptimizeMatrixCellGeneticRunner(GeneticAlgorithmRunner):
    ''''
    Algorithm #3: only mutate the cells that are more distant from
    the ground truth matrix
    '''
    def breed(self):
        # create new children
        new_children = list()
        for individual in self.population:
            for i in range(self.num_children):
                child = self.mutate(individual.dist_matrix)
                new_children.append(child)

        # add to population
        for child in new_children:
            self.add_to_population(child)


def random_pick_dist_matrix_cell(dim_matrix, num_cells_to_pick):
    '''
    Returns N (num_cells_to_picks) indices of cells randomly picked
    Diagonals are NOT returned
    Symmetric cells are NOT returned
    :param dim_matrix:
    :param num_cells_to_pick:
    :return:
    '''
    all_pairwise_combinations = list(itertools.combinations(range(dim_matrix), 2))
    cell_indices = random.sample(all_pairwise_combinations, num_cells_to_pick)
    return cell_indices

class RandomGeneticRunner(GeneticAlgorithmRunner):
    '''
    Algo 1: mutacao: adicionar erro gaussiano (1 célula)
    '''
    def breed(self):
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
        indices = random_pick_dist_matrix_cell(dist_matrix.shape[0], NUM_CELLS_TO_MUTATE)
        child_matrix = np.copy(dist_matrix)
        for i, j in indices:
            noise = np.random.normal(loc=0.0, scale=SIGMA)
            child_matrix[i,j] += noise
            child_matrix[j,i] += noise

        return child_matrix


class MoveWeightsGeneticRunner(GeneticAlgorithmRunner):
    '''
    Algo 2: mutacao: mover peso de uma celula da matriz pra outro (1 celula)
    '''
    def breed(self):
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
        indices = random_pick_dist_matrix_cell(dist_matrix.shape[0], NUM_CELLS_TO_MUTATE*2)
        child_matrix = np.copy(dist_matrix)
        subtracting = True # flag switches between subtracting and summing
        for i, j in indices:
            noise = np.random.normal(loc=0.0, scale=SIGMA)
            if subtracting:
                child_matrix[i,j] -= noise
                child_matrix[j,i] -= noise
                subtracting = False
            else:
                child_matrix[i,j] += noise
                child_matrix[j,i] += noise
                subtracting = True

        return child_matrix


class CrossoverAverageGeneticRunner(GeneticAlgorithmRunner):
    '''
    Algo 4: crossover: media par-a-par de 2 matrizes (Matriz1 + Matriz2 / 2)
    '''
    def breed(self):
        all_pairwise_combinations = list(itertools.combinations(range(self.num_parents), 2))

        # pick N distinct pairs of parents to breed
        couples_indices = random.sample(all_pairwise_combinations, self.num_children*self.num_parents)

        for i, j in couples_indices:
            child = self.reproduce(self.population[i].dist_matrix, self.population[j].dist_matrix)
            child = self.mutate(child)
            self.add_to_population(child)


    def mutate(self, dist_matrix):
        indices = random_pick_dist_matrix_cell(dist_matrix.shape[0], NUM_CELLS_TO_MUTATE)
        child_matrix = np.copy(dist_matrix)
        for i, j in indices:
            noise = np.random.normal(loc=0.0, scale=SIGMA)
            child_matrix[i,j] += noise
            child_matrix[j,i] += noise

        return child_matrix

    def reproduce(self, matrix1, matrix2):
        '''
        Returns child matrix
        :param matrix1:
        :param matrix2:
        :return:
        '''
        return (matrix1 + matrix2) / 2


class CrossoverMergeGeneticRunner(GeneticAlgorithmRunner):
    '''
    Algo 5: crossover: mergear partes de cada 2 matrizes (50%/50%) - rand(50%)
    '''
    def breed(self):
        all_pairwise_combinations = list(itertools.combinations(range(self.num_parents), 2))

        # pick N distinct pairs of parents to breed
        couples_indices = random.sample(all_pairwise_combinations, self.num_children*self.num_parents)

        for i, j in couples_indices:
            child = self.reproduce(self.population[i].dist_matrix, self.population[j].dist_matrix)
            child = self.mutate(child)
            self.add_to_population(child)

        print(len(self.population))
    def reproduce(self, matrix1, matrix2):
        '''
        Returns child matrix
        :param matrix1:
        :param matrix2:
        :return:
        '''
        dim_matrix = matrix1.shape[0]
        couples_indices = random_pick_dist_matrix_cell(dim_matrix, int((dim_matrix**2 - dim_matrix)/4))

        # merge 50%/50% matrix 1 and 2
        # overwrite 50% of the cells of the child with values from matrix2
        child = np.copy(matrix1)
        for i, j in couples_indices:
            child[i, j] = matrix2[i, j]
            child[j, i] = matrix2[j, i]

        return child

    def mutate(self, dist_matrix):
        indices = random_pick_dist_matrix_cell(dist_matrix.shape[0], NUM_CELLS_TO_MUTATE)
        child_matrix = np.copy(dist_matrix)
        for i, j in indices:
            noise = np.random.normal(loc=0.0, scale=SIGMA)
            child_matrix[i,j] += noise
            child_matrix[j,i] += noise

        return child_matrix


def main(args):
    directory_path = args.in_data
    directory = os.fsencode(directory_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("additive.noisy.npy"):
            dataset_matrix = np.load(os.path.join(directory_path, filename))

            if args.algo == 1:
                runner = RandomGeneticRunner(ground_truth_matrix=dataset_matrix,
                                             num_iterations=NUM_ITERATIONS,
                                             num_children=NUM_CHILDREN,
                                             num_parents=NUM_PARENTS
                                             )
            elif args.algo == 2:
                runner = MoveWeightsGeneticRunner(ground_truth_matrix=dataset_matrix,
                                             num_iterations=NUM_ITERATIONS,
                                             num_children=NUM_CHILDREN,
                                             num_parents=NUM_PARENTS
                                             )
            elif args.algo == 3:
                runner = OptimizeMatrixCellGeneticRunner(ground_truth_matrix=dataset_matrix,
                                                         num_iterations=NUM_ITERATIONS,
                                                         num_children=NUM_CHILDREN,
                                                         num_parents=NUM_PARENTS
                                                         )
            elif args.algo == 4:
                runner = CrossoverAverageGeneticRunner(ground_truth_matrix=dataset_matrix,
                                             num_iterations=NUM_ITERATIONS,
                                             num_children=NUM_CHILDREN,
                                             num_parents=NUM_PARENTS
                                             )
            elif args.algo == 5:
                runner = CrossoverMergeGeneticRunner(ground_truth_matrix=dataset_matrix,
                                             num_iterations=NUM_ITERATIONS,
                                             num_children=NUM_CHILDREN,
                                             num_parents=NUM_PARENTS
                                             )
            else:
                raise Exception('You must pass argument --algo=<index>')

            results = runner.run()
            results['dataset'] = directory_path
            results['matrix'] = filename
            results['algo'] = args.algo
            with open(args.out, 'a') as f:
                results.to_csv(f, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic algorithms to create phylogenetic trees.')
    parser.add_argument('--iter', type=int, help='runs the algorithm for this number of iterations')
    parser.add_argument('--in-data', help='runs genetic algorithms using source data')
    parser.add_argument('--out', help='path to CSV containing the results')
    parser.add_argument('--algo', type=int)
    global NUM_ITERATIONS
    args = parser.parse_args()
    NUM_ITERATIONS = args.iter
    main(args)
