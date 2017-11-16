import numpy as np
import random
import os
import time
import sys

def get_edges(valid_tree):
    row = []
    col = []
    for key, values in valid_tree.items():
        row.append(key)
        col.append(values[0])
        row.append(key)
        col.append(values[1])

    nodes = np.unique(row + col)
    indices = np.argsort(nodes)
    new_index =  dict(zip(nodes, indices))
    row = map(lambda x: new_index[x], row)
    col = map(lambda x: new_index[x], col)

    return row, col


def get_matrix_dist(valid_tree, num_leafs, lowest_weight=1, highest_weight=10):
    '''
    Based on a valid tree, create the additive matrix + some gaussian noise.
    :param valid_tree: dictionary where keys are nodes and the value
    is an array with the 2 child nodes
    :param lowest_weight:
    :param highest_weight:
    :return:
    '''
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra, shortest_path

    row, col = get_edges(valid_tree)

    # convert indexes
    wgt = np.random.randint(lowest_weight, highest_weight, size=len(row))

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
    noisy_matrix = dist_matrix + noise
    dist_matrix = dist_matrix.astype(int)
    noisy_matrix = noisy_matrix.astype(int)

    print('------- noisy matrix --------')
    print(noisy_matrix)

    print('------- dist matrix ------')
    print(dist_matrix)

    return dist_matrix, noisy_matrix, row, col, wgt




def plot_tree(valid_tree, input_amount, filename):
    '''
    Given a valid tree, it will plot a graph image
    and save it as jpg in the filename specified
    :param valid_tree: dictionary where keys are nodes and the value
    is an array with the 2 child nodes
    :param filename: 
    :return: 
    '''
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.Graph()
    _, _, row, col, wgt = get_matrix_dist(valid_tree, input_amount)
    G.add_nodes_from(np.unique(row + col))
    G.add_weighted_edges_from(zip(row, col, wgt))

    'save using the standard layout by networkx (sometimes not so readable but' \
    'can be better than spring sometimes)'
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=450, font_size=11)
    plt.savefig(filename)
    plt.gcf().clear()

    'save using spring layout (this seems to be more readable)'
    nx.draw_spring(G, with_labels=True, node_color='lightblue', node_size=450, font_size=11)
    plt.savefig(filename.replace('.jpg','.spring.jpg'))
    plt.gcf().clear()
