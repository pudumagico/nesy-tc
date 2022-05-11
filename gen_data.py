import argparse
import csv
import os
import shutil
import sys
import random

from tqdm import tqdm
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx_nodes, draw_networkx_edges
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def draw_node_bbox(ax, pos, size):
    for key in pos:
        rect = patches.Rectangle(pos[key], size, size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    return


def draw_edge_bbox(ax, gr, pos):
    pass


def save_graph(gr, i, output_folder, output_format):

    if output_format == 'image':
        # nx.draw(gr, node_size=500, with_labels=True)
        cf = plt.gcf()
        cf.set_facecolor("w")
        cf.set_figheight(10)
        cf.set_figwidth(10)
        ax = cf.add_axes((0, 0, 1, 1))

        # plt.figure(figsize=(5, 5), dpi=640)'
        node_size = 500

        pos = nx.drawing.random_layout(gr)
        nodes = nx.draw_networkx_nodes(gr, pos, node_size=node_size)
        edges = nx.draw_networkx_edges(gr, pos)

        # draw_node_bbox(ax, pos, node_size)

        ax.set_axis_off()

        print('nodes', nodes.__dict__)
        print('edges', edges)
        for edge in edges:
            print(edge.__dict__)

        for i in list(gr):
            print(i, pos[i])
        # print(str(plt))
        plt.show()
        plt.savefig(output_folder + '/' + i + ".jpg",
                    format="JPG", bbox_inches='tight')
        plt.clf()
    else:
        nx.write_adjlist(gr, output_folder + '/' + i + ".txt")


def main(dataset_size, nodes, connectivity, output_folder, split, output_format, prop):

    output_path = os.path.abspath(os.getcwd()) + '/' + output_folder
    train_path = output_folder + '/train'
    test_path = output_folder + '/test'

    if output_format == 'image':
        extension = '.jpg'
    else:
        extension = '.txt'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    train_file = open(output_path + '/train.csv', 'w')
    train_writer = csv.writer(train_file)

    test_file = open(output_path + '/test.csv', 'w')
    test_writer = csv.writer(test_file)

    classes = {}

    i = 0
    with tqdm(total=dataset_size) as pbar:
        while i < dataset_size:
            adj_matrix = np.random.rand(nodes, nodes)
            for row in range(len(adj_matrix)):
                for column in range(len(adj_matrix[row])):
                    if adj_matrix[row][column] > connectivity:
                        adj_matrix[row][column] = 1
                    else:
                        adj_matrix[row][column] = 0

            rows, cols = np.where(adj_matrix == 1)
            edges = zip(rows.tolist(), cols.tolist())
            gr = nx.DiGraph()
            gr.add_edges_from(edges)

            if prop == 'transitive_closure':
                tc = nx.transitive_closure(gr)
                if gr.edges() != tc.edges():
                    save_graph(tc, str(i),
                               output_folder, output_format)
                    classes[i] = 1
                    save_graph(gr, str(i+1),
                               output_folder, output_format)
                    classes[i+1] = 0
                    i += 2
                    pbar.update(2)

    data = [str(i) + extension for i in range(dataset_size)]
    random.shuffle(data)
    train_data = data[:int((len(data)+1)*split)]
    test_data = data[int((len(data)+1)*split):]

    outcomes = [0, 1]

    for image in train_data:
        shutil.move(output_path + '/' + image, train_path + '/' + image)
        train_writer.writerow([image, classes[int(image.split('.')[0])]])

    for image in test_data:
        shutil.move(output_path + '/' + image, test_path + '/' + image)
        test_writer.writerow([image, classes[int(image.split('.')[0])]])

    train_file.close()
    test_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Random Graphs Generator + Transitive Closure')

    parser.add_argument('-ds', '--dataset_size', type=int, metavar='int',
                        default=10, help='number of graphs to produce.')
    parser.add_argument('-n', '--nodes', type=int,
                        metavar='int', default=4, help='number of nodes.')
    # parser.add_argument('-k', '--columns', type=int, metavar='int', default=4, help='n adjacency matrix columns')
    parser.add_argument('-c', '--connectivity', type=float, metavar='float',
                        default=0.8, help='connectivity of the graph. must be between 0 and 1.')
    parser.add_argument('-o', '--output_folder', type=str, metavar='str',
                        default='data', help='folder where the output is saved.')
    parser.add_argument('-s', '--train_test_split', type=float, metavar='float',
                        default='0.8', help='split percentage of the total as training examples.')
    parser.add_argument('-of', '--output_format', type=str, metavar='string',
                        default='image', help='Save images or adjacency matrices.')
    parser.add_argument('-p', '--property', type=str, metavar='string',
                        default='transitive_closure', help='property of the graphs to be generated as positive examples.')

    args = parser.parse_args()

    main(dataset_size=args.dataset_size, nodes=args.nodes, connectivity=args.connectivity,
         output_folder=args.output_folder, split=args.train_test_split, output_format=args.output_format, prop=args.property)
