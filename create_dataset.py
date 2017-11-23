import numpy as np
import random
import os
import time
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, shortest_path
import networkx as nx

def get_edges_from_valid_tree(valid_tree):
    nodes_edge_right = []
    nodes_edge_left = []
    for key, values in valid_tree.items():
        nodes_edge_right.append(key)
        nodes_edge_left.append(values[0])
        nodes_edge_right.append(key)
        nodes_edge_left.append(values[1])

    nodes = np.unique(nodes_edge_right + nodes_edge_left)
    indices = np.argsort(nodes)
    new_index =  dict(zip(nodes, indices))
    nodes_edge_right = [new_index[x] for x in nodes_edge_right]
    nodes_edge_left = [new_index[x] for x in nodes_edge_left]

    return nodes_edge_right, nodes_edge_left

def get_edges_from_weighted_edges(edges):
    '''
    Convert format of map[(node1, node2)]->weight to 3 separate lists:
    nodes_edge_right+nodes_edge_left - represent the edges
    edge_weights = the weights for each edge (this matches with the index in
    nodes_edge_right and nodes_edge_left)
    This function also normalizes the node labels to be in sorted order
    '''
    nodes_edge_right = []
    nodes_edge_left = []
    edge_weights = []
    for key, value in edges.items():
        nodes_edge_right.append(key[0])
        nodes_edge_left.append(key[1])
        edge_weights.append(value)

    nodes = np.unique(nodes_edge_right + nodes_edge_left)
    indices = np.argsort(nodes)
    new_index = dict(zip(nodes, indices))
    nodes_edge_right = [new_index[x] for x in nodes_edge_right]
    nodes_edge_left = [new_index[x] for x in nodes_edge_left]

    return nodes_edge_right, nodes_edge_left, edge_weights


def get_matrix_dist_from_weighted_edges(edges, k):
    '''
    Returns distance matrix size (k,k)
    :param edges: map[(node1, node2)]->weight
    :param k: number of biological objects
    :return: distance matrix size (k,k)
    '''
    nodes_edge_right, nodes_edge_left, edge_weights = get_edges_from_weighted_edges(edges)
    dist_matrix = get_matrix_dist(nodes_edge_right, nodes_edge_left, edge_weights)
    dist_matrix = dist_matrix[0:k, 0:k]
    return dist_matrix


def get_matrix_dist(nodes_edge_right, nodes_edge_left, edge_weights):
    '''
    Calculate distance matrix
    :param weighted_edges: tuples of (n1, n2, w) where n1 and n2 are nodes
    and w is the weight of the edge between those nodes
    :return: return a distance matrix in numpy format
    '''
    num_nodes = len(set(nodes_edge_right + nodes_edge_left))
    graph = csr_matrix((edge_weights, (np.array(nodes_edge_right), np.array(nodes_edge_left))), shape=(num_nodes, num_nodes))
    dist_matrix = shortest_path(graph, directed=False)
    return dist_matrix


def create_artificial_matrix_dist(valid_tree, num_leafs, lowest_weight=1, highest_weight=10, stdev=1):
    '''
    Based on a valid tree, create the additive matrix + some gaussian noise.
    :param valid_tree: dictionary where keys are nodes and the value
    is an array with the 2 child nodes
    :param lowest_weight:
    :param highest_weight:
    :return:
    '''
    # create distance matrix with random weights
    nodes_edge_right, nodes_edge_left = get_edges_from_valid_tree(valid_tree)
    edge_weights = np.random.randint(lowest_weight, highest_weight, size=len(nodes_edge_right))
    dist_matrix = get_matrix_dist(nodes_edge_right, nodes_edge_left, edge_weights)

    # only the leaf nodes which are biological objects
    dist_matrix = dist_matrix[0:num_leafs, 0:num_leafs]

    # add gaussian noise
    noise = np.rint(np.random.normal(0, stdev, dist_matrix.shape))
    noise = np.tril(noise) + np.tril(noise, -1).T
    np.fill_diagonal(noise, 0)
    noisy_matrix = dist_matrix + noise
    dist_matrix = dist_matrix.astype(int)
    noisy_matrix = noisy_matrix.astype(int)

    return dist_matrix, noisy_matrix, nodes_edge_right, nodes_edge_left, edge_weights


def plot_tree(valid_tree, input_amount, filename):
    '''
    Given a valid tree, it will plot a graph image
    and save it as jpg in the filename specified
    :param valid_tree: dictionary where keys are nodes and the value
    is an array with the 2 child nodes
    :param filename: 
    :return: 
    '''
    # import pygraphviz as pgv
    #
    # G = pgv.AGraph(splines=False)
    #
    # edges = get_edges_from_valid_tree(valid_tree)
    # for edge in edges.keys():
    #     G.add_node(edge[0], shape='circle' if edge[0] < input_amount else 'point')
    #     G.add_node(edge[1], shape='circle' if edge[1] < input_amount else 'point')
    #
    # for edge, weight in edges.items():
    #     G.add_edge(edge[0], edge[1], label=str(weight), color='gray')
    #
    # G.draw(filename, prog='dot')  # draw png
    pass
