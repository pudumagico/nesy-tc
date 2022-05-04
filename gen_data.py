import argparse
import os
import sys

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def save_graph_with_labels(i, adjacency_matrix, output_folder):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, with_labels=True)
    # plt.show()
    plt.savefig(output_folder + '/' + i + ".png", format="PNG", bbox_inches='tight')
    plt.clf()

def main(dataset_size, nodes, connectivity, output_folder):

    output_path = os.path.abspath(os.getcwd()) + '/' + output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(dataset_size):
        adj_matrix = np.random.rand(nodes, nodes)
        for row in range(len(adj_matrix)):
            for column in range(len(adj_matrix[row])):
                if adj_matrix[row][column] > connectivity:
                    adj_matrix[row][column] = 1
                else:
                    adj_matrix[row][column] = 0

    # print(adj_matrix)
        save_graph_with_labels(str(i), adj_matrix, output_folder)
        # graph = nx.DiGraph(adj_matrix)
        # transitive_closure = nx.transitive_closure(graph)
    # print(transitive_closure.edges())



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Random Graphs Generator + Transitive Closure')

    parser.add_argument('-ds', '--dataset_size', type=int, metavar='int', default=10, help='number of graphs to produce.')
    parser.add_argument('-n', '--nodes', type=int, metavar='int', default=4, help='number of nodes.')
    # parser.add_argument('-k', '--columns', type=int, metavar='int', default=4, help='n adjacency matrix columns')
    parser.add_argument('-c', '--connectivity', type=float, metavar='int', default=0.8, help='connectivity of the graph. must be between 0 and 1.')
    parser.add_argument('-o', '--output_folder', type=str, metavar='str', default='./data', help='folder where the output is saved.')


    args = parser.parse_args()

    main(dataset_size = args.dataset_size, nodes = args.nodes, connectivity=args.connectivity, output_folder=args.output_folder)

