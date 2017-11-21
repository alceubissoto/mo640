# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import time
import sys
import copy
from tqdm import tqdm

def upgma(matrix, input_amount):
    new_ind = {}
    new_ind['genotype'] = []
    min_value = 10000 # infinito positivo
    updt_mtx = matrix
    next_node = input_amount

    heights = {}
    for i in range(input_amount):
        heights[i] = 0
    #print(heights)

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

    return heights, new_ind

def active_nodes(ind, input_amount):
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


def evaluate_tree(actv, heights, input_amount):
    matrix = np.zeros((input_amount, input_amount))
    evaluated = {}
    for i in range(input_amount):
        evaluated[i] = [i]

    for item in sorted(actv.items()):
        evaluated[item[0]] = []
        evaluated[item[0]].extend(evaluated[item[1][0]])
        evaluated[item[0]].extend(evaluated[item[1][1]])

        for obj in evaluated[item[0]]:
            for obj_2 in evaluated[item[0]]:
                if (obj != obj_2) and matrix[obj, obj_2] == 0:
                    matrix[obj, obj_2] = heights[item[0]]*2

    return np.array(matrix)

def compare_matrix(mtx1, mtx2):
    differences = 0
    assert(mtx1.shape == mtx2.shape)
    assert(mtx1.shape[0] == mtx1.shape[1])
    for i in range(mtx1.shape[0]):
        for j in range(mtx1.shape[1]):
            differences += (mtx1[i,j] - mtx2[i,j])**2

    return differences

def mutate(heights, input_amount):
    new_heights = copy.deepcopy(heights)
    candidate_keys = [k for k,_ in new_heights.items() if k >= input_amount]
    choice = random.choice(candidate_keys)
    random_number = random.gauss(0,1)
    new_heights[choice] += random_number

    return new_heights

def reproduction(original_mtx, input_amount, pop_size, iterations):
    orig_modified = np.hstack((np.array(range(input_amount)).reshape((input_amount, 1)), original_mtx))
    heights, first_tree = upgma(orig_modified, input_amount)
    orig_heights = copy.deepcopy(heights)
    actv = active_nodes(first_tree, input_amount)
    upga_mtx = evaluate_tree(actv, heights, input_amount)
    upgma_fitness = compare_matrix(original_mtx, upga_mtx)
    #print("UPGMA Fitness: ", upgma_fitness)
    population = []
    best_fitness = upgma_fitness
    for iteration in tqdm(range(iterations)):
        for i in range(pop_size):
            individual = {}
            individual['heights'] = heights
            gen_mtx = evaluate_tree(actv, heights, input_amount)
            individual['fitness'] = compare_matrix(original_mtx, gen_mtx)
            #print(gen_mtx)
            population.append(individual)
            if(individual['fitness'] < best_fitness):
                best_fitness = individual['fitness']

            # Mutation
            heights = mutate(orig_heights, input_amount)

        for i in range(pop_size):
            if population[i]['fitness'] == best_fitness:
                tmp = []
                tmp.append(population[i])
                break
        population = tmp

        orig_heights = copy.deepcopy(population[0]['heights'])

    return population[0], actv, first_tree
