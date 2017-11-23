# -*- coding: utf-8 -*-
from __future__ import print_function
from dataset import get_matrix_dist_from_weighted_edges, create_artificial_matrix_dist, plot_tree
from plotTree import plotTree
import numpy as np
import random
import os
import time
import sys
import argparse
import copy
from neighbor_join import NeighborJoinRunner
from tqdm import tqdm
import weight_cgp
import pandas as pd

# Seria interessante receber os parametros antes da execucao
ind_size = 100   # Size of every individual
input_amount = 20 # Amount of inputs
func_amount = 2  # Amount of functions STILL NEED TO DEFINE THESE
pop_size = 5     # Population Size
# Talvez as operacoes devessem ser já formatos de arvores (uniao de varios nos)..

def create_new_individual():
    new_ind = {}
    new_ind['genotype'] = []
    possible_nodes = list(range(0, input_amount)) # Possible nodes to be add as inputs / output
    operators = list(range(0, func_amount))
    last = input_amount -1;
    for i in range(0, ind_size):
        #tmp_value = random.choice(possible_nodes)
        new_ind['genotype'].append(random.choice(possible_nodes))
        #if tmp_value < input_amount:
        #    possible_values.pop(tmp_value)
        new_ind['genotype'].append(random.choice(possible_nodes))
        new_ind['genotype'].append(random.choice(operators))
        last+=1
        possible_nodes.append(last)
    new_ind['output'] = random.choice(possible_nodes[input_amount:])
    #new_ind['fitness'] = evaluate(new_ind)
    return new_ind

# Separa os nós ativos presentes.
def active_nodes(ind):
    out = ind['output']
    gen = ind['genotype']
    to_eval=[]
    to_eval.append(out)
    evaluated = list(range(0, input_amount))
    used_nodes = {}
    inicial_position = input_amount
    while(len(to_eval) > 0):
        if(to_eval[0] in evaluated):
            to_eval.pop(0)
        else:
            tmp = []
            for i in range(0,3):
                value = gen[(to_eval[0]-inicial_position)*3 + i]
                tmp.append(value)
                if((value not in evaluated) and (i!=2)):
                    to_eval.append(value)
            used_nodes[to_eval[0]] = tmp
            evaluated.append(to_eval[0])
    return used_nodes


def neighbor_join(matrix):
    runner = NeighborJoinRunner(matrix, input_amount)
    return runner.get_individual(), runner.get_dist_matrix()

def upgma(matrix):
    new_ind = {}
    new_ind['genotype'] = []
    min_value = 10000 # positive infinite
    updt_mtx = matrix
    next_node = input_amount

    # initialize heights
    heights = {}
    for i in range(input_amount):
        heights[i] = 0

    while (updt_mtx.shape[0] > 1):
        min_value = 10000 # positive infinite
        # Search for the closest objects

        for row in range(updt_mtx.shape[0]):
            for column in range(1, updt_mtx.shape[1]):
                if (updt_mtx[row, column] < min_value) and (updt_mtx[row, column] > 0):
                    min_value = updt_mtx[row, column]
                    C1_idx = updt_mtx[row, 0]
                    C1_row = row
                    C1_column = column
                    C2_idx = updt_mtx[column-1, 0]
                    C2_row = column-1
                    C2_column = row+1
                    C1 = updt_mtx[row, :]
                    C2 = updt_mtx[column-1, :]

        # Evaluate height of the new node
        heights[next_node] = min_value/2

        # Delete the rows and columns with respect to the selected objects
        if C1_column > C2_column:
            updt_mtx = np.delete(updt_mtx, C1_column, 1)
            updt_mtx = np.delete(updt_mtx, C2_row, 0)
            updt_mtx = np.delete(updt_mtx, C2_column, 1)
            updt_mtx = np.delete(updt_mtx, C1_row, 0)
            C1 = np.delete(C1, C1_column)
            C1 = np.delete(C1, C2_column)
            C2 = np.delete(C2, C1_column)
            C2 = np.delete(C2, C2_column)
        else:
            updt_mtx = np.delete(updt_mtx, C2_column, 1)
            updt_mtx = np.delete(updt_mtx, C1_row, 0)
            updt_mtx = np.delete(updt_mtx, C1_column, 1)
            updt_mtx = np.delete(updt_mtx, C2_row, 0)
            C1 = np.delete(C1, C2_column)
            C1 = np.delete(C1, C1_column)
            C2 = np.delete(C2, C2_column)
            C2 = np.delete(C2, C1_column)

#        print(updt_mtx, min_value)

        # Evaluate the new row/column values
        D_row = np.array(range(updt_mtx.shape[1]))
        D_row[0] = next_node
        C1 = np.ravel(C1)
        C2 = np.ravel(C2)

        for i in range(1, len(D_row)):
            D_row[i] = (C1[i]+C2[i])/2

        # Concatenate the new row/column to the matrix
        D_column = np.matrix(np.append(D_row[1:], 0)).transpose()
        updt_mtx = np.append(updt_mtx, [D_row], axis=0)
        updt_mtx = np.append(updt_mtx, D_column, axis=1)

        new_ind['genotype'].append(C1_idx)
        new_ind['genotype'].append(C2_idx)
        new_ind['genotype'].append(0)

        next_node = next_node + 1

    new_ind['output'] = next_node -1

    #new_ind = fill_individual(new_ind)

    return heights, new_ind


def evaluate_upgma_matrix(actv, heights):
    matrix = np.zeros((input_amount, input_amount))
    evaluated = {}
    #print(actv)
    #print(heights)
    for i in range(input_amount):
        evaluated[i] = [i]

    for item in sorted(actv.items()):
        evaluated[item[0]] = []
        evaluated[item[0]].extend(evaluated[item[1][0]])
        evaluated[item[0]].extend(evaluated[item[1][1]])

        for obj in evaluated[item[0]]:
            for obj_2 in evaluated[item[0]]:
                #print("ITEM[0]:", item[0])
                #print("HEIGHTS:", heights)
                if (obj != obj_2) and matrix[obj, obj_2] == 0:
                    matrix[obj, obj_2] = heights[item[0]]*2

    return np.array(matrix)


def fill_individual(ind):
    new_ind = {}
    new_ind['genotype'] = []
    out = ind['output']
    tmp_ind = create_new_individual() # individual we are merging our ultrametric tree.
    factor = int(ind_size/out) # Factor by which the keys will be multiplied
    active = active_nodes(ind)
    new_active = {}
    # New values for the actual nodes of the tree
    for key, value in active.items():
        for i in range(len(value)-1):
            if (value[i] > input_amount-1):
                value[i] = value[i]*factor
        new_active[key*factor] = value
#    print("ACTIVE:", new_active)

    # New individual formation
    for i in range(ind_size):
        if(i+input_amount in new_active.keys()):
            new_ind['genotype'].append(new_active[i+input_amount][0])
            new_ind['genotype'].append(new_active[i+input_amount][1])
            new_ind['genotype'].append(new_active[i+input_amount][2])
        else:
            new_ind['genotype'].append(tmp_ind['genotype'][i*3])
            new_ind['genotype'].append(tmp_ind['genotype'][i*3+1])
            new_ind['genotype'].append(tmp_ind['genotype'][i*3+2])
    new_ind['output'] = ind['output']*factor
    return new_ind


def is_valid_graph(active):
    count = 0
    for _, value in active.items():
        for i in range(len(value)-1):
            if value[i] < input_amount:
                count = count + 1
    if count == input_amount:
        return True
    else:
        return False


def get_binary_tree(valid_graph):
    root = max(valid_graph.keys())
    # set input in order
    inputs = set(range(input_amount))
    for key, values in valid_graph.items():
        if values[0] < input_amount:
            if values[0] in inputs:
                inputs.remove(values[0])
            else:
                values[0] = inputs.pop()
        if values[1] < input_amount:
            if values[1] in inputs:
                inputs.remove(values[1])
            else:
                values[1] = inputs.pop()

    visited = set()
    valid_tree = dict()

    # build binary tree, assumes that tree has at least 2 inputs
    recursive_get_binary_tree(root, valid_graph, visited, valid_tree)
    # print('Valid tree generated', valid_tree)
    return valid_tree


def recursive_get_binary_tree(cur, valid_graph, visited, valid_tree):
    if cur in visited:
        return -1

    visited.add(cur)

    # If input, return itself
    if cur < input_amount:
        return cur

    left = valid_graph[cur][0]
    right = valid_graph[cur][1]
    node_with_input_left = recursive_get_binary_tree(left, valid_graph, visited, valid_tree)
    node_with_input_right = recursive_get_binary_tree(right, valid_graph, visited, valid_tree)

    if node_with_input_left >= 0 and node_with_input_right >= 0:
        valid_tree[cur] = [node_with_input_left, node_with_input_right]
        return cur

    return node_with_input_left if node_with_input_left >= 0 else node_with_input_right

def find_edge_weights_ultrametric(valid_tree, matrix):
    #print(valid_tree)
    #print(matrix)
    # Dict with edges and weights
    edges = dict()
    #print('Valid tree', valid_tree)

    while matrix.shape[0] > 2:

        nodes = set(matrix[:, 0])

        # Find two adjacent nodes in set nodes
        adjacencies = []
        for parent, children in valid_tree.items():
            if children[0] in nodes and children[1] in nodes:
                adjacencies.append((parent, children[0], children[1]))

        adjacent = random.choice(adjacencies)
        parent, a, b = adjacent

        # Find coordinates of a, b and c
        row_a = list(matrix[:, 0]).index(a)
        row_b = list(matrix[:, 0]).index(b)
        col_a = row_a + 1
        col_b = row_b + 1

        # Find another node c in set of nodes, that is not one of the two adjacent (a,b)
        # such that minimizes |dist(a,c) - dist(b,c)|
        mindist = 10000000 # big number
        for row, node in enumerate(matrix[:, 0]):
            if node not in (a, b):
                row_c, col_c, c = row, row+1, node
                break

        # Compute the edge weight of the adjacent nodes to their parent
        ap = max(0, (matrix[row_a, col_b] + matrix[row_a,col_c] - matrix[row_b, col_c])/2)
        bp = max(0, (matrix[row_b, col_a] + matrix[row_b,col_c] - matrix[row_a, col_c])/2)
        edges[(a, parent)] = (ap + bp)/2
        edges[(b, parent)] = (ap + bp)/2

        # Now replace the row and column of 'a' with its parent
        matrix[row_a, 0] = parent
        matrix[row_a, 1:] = [max(0, (dist_a - ap +  dist_b - bp)/2) for dist_a, dist_b in zip(matrix[row_a, 1:], matrix[row_b, 1:])]
        #matrix[row_a, 1:] = [max(0, (dist - ap)) for dist in matrix[row_a, 1:]]
        matrix[:, col_a] = matrix[row_a, 1:]
        # and remove the row and column of 'b'
        matrix = np.delete(matrix, row_b, 0)
        matrix = np.delete(matrix, col_b, 1)
        #print(matrix)

    # Add distance between last nodes and remove root
    #print('Final matrix:', matrix)
    edges[(matrix[0,0], matrix[1,0])] = matrix[0,2]
    #print(edges)
    #plotTree(edges, input_amount)

    # format: key (edge1, edge2) and value (weight)
    print(edges)
    return edges


def find_edge_weights(valid_tree, matrix):
    #print(valid_tree)
    #print(matrix)
    # Dict with edges and weights
    edges = dict()
    #print('Valid tree', valid_tree)

    while matrix.shape[0] > 2:

        nodes = set(matrix[:, 0])

        # Find two adjacent nodes in set nodes
        adjacencies = []
        for parent, children in valid_tree.items():
            if children[0] in nodes and children[1] in nodes:
                adjacencies.append((parent, children[0], children[1]))

        adjacent = random.choice(adjacencies)
        parent, a, b = adjacent

        # Find coordinates of a, b and c
        row_a = list(matrix[:, 0]).index(a)
        row_b = list(matrix[:, 0]).index(b)
        col_a = row_a + 1
        col_b = row_b + 1

        # Find another node c in set of nodes, that is not one of the two adjacent (a,b)
        # such that minimizes |dist(a,c) - dist(b,c)|
        mindist = 10000000 # big number
        for row, node in enumerate(matrix[:, 0]):
            if node not in (a, b):
                row_c, col_c, c = row, row+1, node
                break

        # Compute the edge weight of the adjacent nodes to their parent
        ap = max(0, (matrix[row_a, col_b] + matrix[row_a,col_c] - matrix[row_b, col_c])/2)
        bp = max(0, (matrix[row_b, col_a] + matrix[row_b,col_c] - matrix[row_a, col_c])/2)
        edges[(a, parent)] = ap
        edges[(b, parent)] = bp

        # Now replace the row and column of 'a' with its parent
        matrix[row_a, 0] = parent
        #matrix[row_a, 1:] = [max(0, (dist_a - ap +  dist_b - bp)/2) for dist_a, dist_b in zip(matrix[row_a, 1:], matrix[row_b, 1:])]
        matrix[row_a, 1:] = [max(0, (dist - ap)) for dist in matrix[row_a, 1:]]
        matrix[:, col_a] = matrix[row_a, 1:]
        # and remove the row and column of 'b'
        matrix = np.delete(matrix, row_b, 0)
        matrix = np.delete(matrix, col_b, 1)
        #print(matrix)

    # Add distance between last nodes and remove root
    #print('Final matrix:', matrix)
    edges[(matrix[0,0], matrix[1,0])] = matrix[0,2]
    #print(edges)
    #plotTree(edges, input_amount)

    # format: key (edge1, edge2) and value (weight)
    return edges


def mutation(ind): #STILL NEED TO BE ABLE TO MUTATE THE OUTPUT
    gen = copy.deepcopy(ind)
    #print(gen)
    used = active_nodes(gen)
    #print(used)
    used_list = list(used.items())
    key = random.choice(used_list)
    #print(key)
    chosen_item = random.randint(0,1)
    #print(chosen_item)
    gen['genotype'][(key[0]-input_amount)*3+chosen_item] = random.randint(0, key[0]-1)
    return gen


def create_matrix_dist(args):
    '''
    Creates matrix distances in the quantity specified.
    Saves matrix distances as numpy files.
    :param qty: number of matrices to create
    :return: none
    '''
    out_data = args.out_data
    input_amount = args.matrix_count
    qty = args.qty

    matrix = np.array([[0, 0, 12, 12, 12, 12],
                       [1, 12, 0, 4, 6, 6],
                       [2, 12, 4, 0, 6, 6],
                       [3, 12, 6, 6, 0, 2],
                       [4, 12, 6, 6, 2, 0]])

    _, new_ind = upgma(matrix)
    new_ind = fill_individual(new_ind)

    mutated = mutation(new_ind)
    for i in range(qty):
        while True:
            mutated = mutation(mutated)
            active_mut = active_nodes(mutated)
            if is_valid_graph(active_mut):
                valid_tree = get_binary_tree(active_mut)
                dist_matrix, noisy_matrix, _, _, _ = create_artificial_matrix_dist(valid_tree,
                                                                                   num_leafs=args.num_leafs,
                                                                                   highest_weight=args.max_weight,
                                                                                   stdev=args.stdev)

                if not os.path.exists(out_data):
                    os.makedirs(out_data)

                millis = int(round(time.time() * 1000))
                np.save(os.path.join(out_data, str(millis) + '.additive.npy'), dist_matrix)
                np.save(os.path.join(out_data, str(millis) + '.additive.noisy.npy'), noisy_matrix)
                np.savetxt(os.path.join(out_data, str(millis) + '.additive.txt'), dist_matrix, fmt='%d')
                np.savetxt(os.path.join(out_data, str(millis) + '.additive.noisy.txt'), noisy_matrix, fmt='%d')

                plot_tree(valid_tree, num_leafs, os.path.join(out_data, str(millis) + '.jpg'))
                break


def test_upgma_fitness(dataset_matrix):
    k = dataset_matrix.shape[0] # number of biological objects
    indexed_matrix = np.hstack((np.array(range(input_amount)).reshape((input_amount, 1)), dataset_matrix))
    heights, upgma_tree = upgma(indexed_matrix)
    upgma_individual = fill_individual(upgma_tree)
    actv = active_nodes(upgma_tree)
    upgma_matrix = evaluate_upgma_matrix(actv, heights)
    #print("UPGMA MATRIX:", upgma_matrix)
    fitness = get_fitness(upgma_matrix, dataset_matrix)
    return fitness, upgma_individual

def test_neighbor_fitness(dataset_matrix):
    individual, dist_matrix = neighbor_join(dataset_matrix)
    neighbor_individual = fill_individual(individual)
    fitness = get_fitness(dist_matrix, dataset_matrix)
    return fitness, neighbor_individual

def get_fitness(dist_matrix1, dist_matrix2):
    return np.square(dist_matrix1 - dist_matrix2).sum()

def evaluate_ultrametric(individual, dataset_matrix):
    k = dataset_matrix.shape[0] # number of biological objects
    indexed_matrix = np.hstack((np.array(range(input_amount)).reshape((input_amount, 1)), dataset_matrix))
    valid_tree = get_binary_tree(active_nodes(individual))
    edges = find_edge_weights_ultrametric(valid_tree, indexed_matrix)
    dist_matrix = get_matrix_dist_from_weighted_edges(edges, k)
    fitness = get_fitness(dist_matrix, dataset_matrix)

    # Loop 50 times and get best matrix generated for that topology
    # using random order of find_edge_weights
    
    print ('First fitness:', fitness)
    for i in range(50):
        indexed_matrix = np.hstack((np.array(range(input_amount)).reshape((input_amount, 1)), dataset_matrix))
        new_edges = find_edge_weights(valid_tree, indexed_matrix)
        new_dist_matrix = get_matrix_dist_from_weighted_edges(new_edges, k)
        new_fitness = get_fitness(new_dist_matrix, dataset_matrix)
        if new_fitness < fitness:
            edges, dist_matrix, fitness = new_edges, new_dist_matrix, new_fitness
    print ('Last fitness:', fitness)

    #print('**** fitness = ' + str(fitness))
    return fitness

def evaluate(individual, dataset_matrix):
    k = dataset_matrix.shape[0] # number of biological objects
    indexed_matrix = np.hstack((np.array(range(input_amount)).reshape((input_amount, 1)), dataset_matrix))
    valid_tree = get_binary_tree(active_nodes(individual))
    edges = find_edge_weights(valid_tree, indexed_matrix)
    dist_matrix = get_matrix_dist_from_weighted_edges(edges, k)
    fitness = get_fitness(dist_matrix, dataset_matrix)

    # Loop 50 times and get best matrix generated for that topology
    # using random order of find_edge_weights
    
    #print ('First fitness:', fitness)
    for i in range(50):
        indexed_matrix = np.hstack((np.array(range(input_amount)).reshape((input_amount, 1)), dataset_matrix))
        new_edges = find_edge_weights(valid_tree, indexed_matrix)
        new_dist_matrix = get_matrix_dist_from_weighted_edges(new_edges, k)
        new_fitness = get_fitness(new_dist_matrix, dataset_matrix)
        if new_fitness < fitness:
            edges, dist_matrix, fitness = new_edges, new_dist_matrix, new_fitness
    #print ('Last fitness:', fitness)

    #print('**** fitness = ' + str(fitness))
    return fitness

def valid_mutation(individual):
    count = 0
    while True:
        mutated = mutation(individual)
        active_mut = active_nodes(mutated)
        count = count + 1
        if is_valid_graph(active_mut):
            break

    return mutated


def mutate_select(dataset_matrix, individual, num_iterations=100):
    best_fitness = 10000000

    partial_best_fitness = pd.DataFrame(columns=['iteration','gm_time','best_fitness'], index=np.arange(num_iterations))
    population = list()
    population.append(individual)

    # Reproduction
    for iteration in tqdm(range(num_iterations)):
        # Mutation procedure
        for i in range(pop_size - 1):
            mut = valid_mutation(population[0])
            mut['fitness'] = evaluate(mut, dataset_matrix)
            population.append(mut)

        for i in range(pop_size):
            if population[i]['fitness'] < best_fitness:
                best_fitness = population[i]['fitness']


        # Save partial results
        partial_best_fitness.loc[iteration]['iteration'] = iteration
        partial_best_fitness.loc[iteration]['gm_time'] = time.gmtime()
        partial_best_fitness.loc[iteration]['best_fitness'] = best_fitness

        # Keep only the best individual for the next generation
        # print("BEST FITNESS:", best_fitness)
        # print("ITERATION:", iteration)
        # print("POP SIZE:", len(population))
        for i in range(pop_size):
            if population[i]['fitness'] == best_fitness:
                tmp = list()
                tmp.append(population[i])
                break
        population = tmp

    return population, best_fitness, partial_best_fitness


def run_tests(directory_path):
    results = dict()
    directory = os.fsencode(directory_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("additive.npy"):
            print("Testing matrix:", filename)

            dataset_matrix = np.load(os.path.join(directory_path, filename))

            # Neighbor Joining
            fitness_neighbor, neighbor_individual = test_neighbor_fitness(dataset_matrix)
            fitness_score = evaluate(neighbor_individual, dataset_matrix)

            results['neighbor_initial'+filename] = fitness_score
            results['neighbor_pure'+filename] = fitness_neighbor
            neighbor_individual['fitness'] = fitness_score
            population, best_fitness, partial_results = mutate_select(dataset_matrix, neighbor_individual,num_iterations=1)
            assert(population[0]['fitness'] == best_fitness)
            results['neighbor_final'+filename] = population[0]['fitness']
            csv_file_path = os.path.join(directory_path, "neighbor_partial_" + filename.replace('npy', 'csv'))
            partial_results.to_csv(csv_file_path)

            # UPGMA
            fitness_upgma, upgma_individual = test_upgma_fitness(dataset_matrix)
            fitness_score = evaluate_ultrametric(upgma_individual, dataset_matrix)

            results['initial_upgma'+filename] = fitness_score
            results['upgma_pure'+filename] = fitness_upgma
            upgma_individual['fitness'] = fitness_score
            population, best_fitness, partial_results = mutate_select(dataset_matrix, upgma_individual,num_iterations=10)
            assert(population[0]['fitness'] == best_fitness)
            results['final_upgma'+filename] = population[0]['fitness']
            partial_results.to_csv("upgma_partial_" + filename.replace('npy', 'csv'))

            # Mutate only Weights UPGMA
            best, actv_weight, tree_weight = weight_cgp.reproduction(dataset_matrix, input_amount, pop_size, 100)
            gen_mtx = weight_cgp.evaluate_tree(actv_weight, best['heights'], input_amount)
            #print("GENERATED MATIX:", gen_mtx)
            results['final_only_weights_mutation'+filename] = get_fitness(gen_mtx, dataset_matrix)

    print('####################################')
    print('            TEST RESULTS            ')
    print('####################################')
    print('Sample\t\t\t\tScore')
    for sample, score in results.items():
        print(sample + '\t\t' + str(score))


def main(args):
    if args.out_data is not None:
        print('Creating dataset...')
        create_matrix_dist(args)
        print('Dataset written to directory: ' + args.out_data)
    elif args.in_data is not None:
        run_tests(args.in_data)
    else:
        print('Usage information: python cgp.py -h')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic algorithms to create phylogenetic trees.')
    parser.add_argument('--out-data', help='creates dataset')
    parser.add_argument('--num-leafs', help='dimension of the matrices created')
    parser.add_argument('--stdev', help='std dev of the noise')
    parser.add_argument('--max-weight', help='max weight')
    parser.add_argument('--qty', help='number of matrices to create')
    parser.add_argument('--in-data', help='runs genetic algorithms using source data')
    args = parser.parse_args()
    main(args)
