# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import time
import sys

# Seria interessante receber os parametros antes da execucao
ind_size = 100   # Size of every individual
input_amount = 5 # Amount of inputs
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
    while(len(to_eval)>0):
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


def upgma(matrix):
    new_ind = {}
    new_ind['genotype'] = []
    min_value = 10000 # infinito positivo
    updt_mtx = matrix
    next_node = input_amount

    while (updt_mtx.shape[0] > 1):
        min_value = 10000 # infinito positivo
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
    new_ind = fill_individual(new_ind)

    return new_ind


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


def isValidGraph(active):
    count = 0
    for _, value in active.items():
        for i in range(len(value)-1):
            if value[i] < input_amount:
                count = count + 1
    if count == input_amount:
        return True
    else:
        return False


def getBinaryTree(valid_graph):
    root = max(valid_graph.keys())
    # set input in order 
    input_number = 0
    for key, values in valid_graph.items():
        if values[0] < input_amount:
            values[0] = input_number
            input_number += 1
        if values[1] < input_amount:
            values[1] = input_number
            input_number += 1
    
    visited = set()
    valid_tree = dict()

    # build binary tree, assumes that tree has at least 2 inputs
    recursiveGetBinaryTree(root, valid_graph, visited, valid_tree)
    return valid_tree


def recursiveGetBinaryTree(cur, valid_graph, visited, valid_tree):
    if cur in visited:
        return -1 

    visited.add(cur)

    # If input, return itself
    if cur < input_amount:
        return cur

    left = valid_graph[cur][0]
    right = valid_graph[cur][1]
    node_with_input_left = recursiveGetBinaryTree(left, valid_graph, visited, valid_tree)
    node_with_input_right = recursiveGetBinaryTree(right, valid_graph, visited, valid_tree)

    if node_with_input_left >= 0 and node_with_input_right >= 0:
        valid_tree[cur] = [node_with_input_left, node_with_input_right]
        return cur

    return node_with_input_left if node_with_input_left >= 0 else node_with_input_right

def findEdgeWeights(valid_tree, matrix):
    print valid_tree, matrix


def mutation(ind): #STILL NEED TO BE ABLE TO MUTATE THE OUTPUT
    gen = ind['genotype']
    out = ind['output']
    #print(gen)
    used = active_nodes(ind)
    #print(used)
    used_list = list(used.items())
    key = random.choice(used_list)
    #print(key)
    chosen_item = random.randint(0,1)
    #print(chosen_item)
    ind['genotype'][(key[0]-input_amount)*3+chosen_item] = random.randint(0, key[0]-1)
    return ind


def get_matrix_dist(valid_tree, num_leafs, lowest_weight=1, highest_weight=10):
    '''
    Based on a valid tree, create the additive matrix + some gaussian noise.
    :param valid_tree:
    :param num_leafs:
    :param lowest_weight:
    :param highest_weight:
    :return:
    '''
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra, shortest_path

    row = []
    col = []
    for key, values in valid_tree.items():
        row.append(key)
        col.append(values[0])
        row.append(key)
        col.append(values[1])

    # convert indexes
    wgt = list(np.random.randint(lowest_weight, highest_weight, size=len(row)))

    nodes = np.unique(row + col)
    indices = np.argsort(nodes)
    new_index =  dict(zip(nodes, indices))
    row = map(lambda x: new_index[x], row)
    col = map(lambda x: new_index[x], col)

    print('------- nodes ------')
    num_nodes = len(set(row + col))
    print('num nodes = ' + str(num_nodes))

    graph = csr_matrix((wgt, (np.array(row), np.array(col))), shape=(num_nodes, num_nodes))

    print('------- edges ------')
    print('*** Nodes from 0 to ' + str(num_leafs-1) + ' are the leaf nodes***')
    print('   edge      weight')
    print(graph)
    dist_matrix = shortest_path(graph, directed = False)

    # only the leaf nodes which are biological objects
    dist_matrix = dist_matrix[0:num_leafs, 0:num_leafs]

    # add gaussian noise
    noise = np.rint(np.random.normal(0, 1, dist_matrix.shape))
    noise = np.tril(noise) + np.tril(noise, -1).T
    np.fill_diagonal(noise, 0)
    dist_matrix += noise
    print('------- noise --------')
    print(noise)

    print('------- dist matrix ------')
    print(dist_matrix)

    return dist_matrix


def create_matrix_dist(qty=1):
    '''
    Creates matrix distances in the quantity specified.
    Saves matrix distances as numpy files.
    :param qty: number of matrices to create
    :return: none
    '''

    matrix = np.array([[0, 0, 12, 12, 12, 12],
                       [1, 12, 0, 4, 6, 6],
                       [2, 12, 4, 0, 6, 6],
                       [3, 12, 6, 6, 0, 2],
                       [4, 12, 6, 6, 2, 0]])
    new_ind = upgma(matrix)
    mutated = mutation(new_ind)
    for i in range(qty):
        while True:
            mutated = mutation(mutated)
            active_mut = active_nodes(mutated)
            if isValidGraph(active_mut):
                valid_tree = getBinaryTree(active_mut)
                dist_matrix = get_matrix_dist(valid_tree, input_amount)

                directory = 'dataset'
                if not os.path.exists(directory):
                    os.makedirs(directory)

                millis = int(round(time.time() * 1000))
                np.save(os.path.join(directory, str(millis) + '.npy'), dist_matrix)
                break

# def plot_tree(valid_tree):
#     from igraph import *
#     g = Graph()
#     g.add_vertices(3)
#     g.add_edges([(0, 1), (1, 2)])
#     layout = g.layout("kk")
#     plot(g, layout=layout)

def test(matrix):
    '''
    SANDBOX
    For everyone to play around
    :return:
    '''

    '''
        a     b     c     d     e
    a   0    12    12    12    12
    b   12    0     4     6     6
    c   12    4     0     6     6
    d   12    6     6     0     2
    e   12    6     6     2     0

    matrix[1][0] = 12

    matrix = np.array([[0, 0, 12, 12, 12, 12],
                       [1, 12, 0, 4, 6, 6],
                       [2, 12, 4, 0, 6, 6],
                       [3, 12, 6, 6, 0, 2],
                       [4, 12, 6, 6, 2, 0]])
    '''

    matrix = np.hstack((np.array(range(input_amount)).reshape((input_amount, 1)), matrix))
    new_ind = upgma(matrix)
    mutated = mutation(new_ind)

    # experiment = []
    # for i in range(10000):
    count = 0
    while True:
        mutated = mutation(mutated)
        # print(new_ind)
        active_mut = active_nodes(mutated)
        # print(mutated)
        # print(active_mut)
        count = count + 1
        # print("COUNT:", count)
        if (isValidGraph(active_mut)):
            break
    # experiment.append(count)
    #    print(i)
    print('----------------- active mut ---------------------')
    print(active_mut)

    print('----------------- valid tree ---------------------')
    valid_tree = getBinaryTree(active_mut)
    print(valid_tree)

    findEdgeWeights(valid_tree, matrix)
    # experiment = np.array(experiment)
    # print("MEAN:", np.mean(experiment))
    # print("STD:", np.std(experiment))

    # print ("COUNT:", count)
    # print(isValidGraph(used))

    #plot_tree(valid_tree)

    '''
    # input: individuo, matriz de distancias.
    # output: 'fitness'
    def evaluate(ind)


    #input: individuo para ser mutado, chance de mutação.
    #output: novo individuo mutado
    def mutation(ind)


    def create_population():
        pop = []
        for i in range(pop_size):
            pop.append(create_new_individual())
        return pop

    #input: population, numero max de geracoes.
    #output: best individual (solution)
    def reproduction():
        # ordena os individuos
        # Cria 4 mutacoes do mais apto.
        # re-calcula o fitness
        # passa pra proxima geração.
    '''


def main(args):
    if len(args) > 1:
        if args[1] == 'dataset':
            print('creating dataset...')
            create_matrix_dist()
            print('dataset created! look at directory "dataset"')
        else:
            dist_matrix = np.load(args[1])
            print dist_matrix
            test(dist_matrix)
    else:
        print '''Usage:
                 Create distance matrix: 
                    $ python cgp.py dataset
                 
                 Run algorithm with input:
                    $ python cgp.py <path_input>'
                    '''


if __name__ == "__main__":
    main(sys.argv)
