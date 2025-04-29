import pandas as pd
import networkx as nx
import numpy as np

def build_adjacency_matrix(graph):
    num_nodes = len(graph.nodes)
    adjacency_mat = np.zeros((num_nodes, num_nodes))
    node_to_index = {}
    for i, node_id in enumerate(graph.nodes):
        node_to_index[node_id] = i
    
    for node_id in graph.nodes:
        i = node_to_index[node_id]
        neighbors = list(graph.neighbors(node_id))
        for neighbor_id in neighbors:
            j = node_to_index[neighbor_id]
            adjacency_mat[i, j] = 1
    
    return adjacency_mat