import networkx as nx
import scipy.io as sio
import numpy as np
import argparse
import os

output_formats = ["adjlist", "edgelist", "adjmat"]

parser = argparse.ArgumentParser(description="Convert mat files to a different format")

parser.add_argument('--dataset', help="Name for the generated dataset file")
parser.add_argument('--input', help="Network file path")
parser.add_argument('--format', help="Required output format",
                                choices=output_formats)
parser.add_argument('--output', help="Output file directory")

args = parser.parse_args()

IP_FILE = args.input
FORMAT  = args.format
OP_FILE = os.path.join(args.output, args.dataset) + '.' + FORMAT
LABEL_FILE = os.path.join(args.output, args.dataset) + '.npy'

mat = sio.loadmat(IP_FILE)

functions = {
            'adjlist': nx.write_adjlist,
            'edgelist': nx.write_edgelist,
            }


network = mat['network']
labels = mat['group']

G = nx.convert_matrix.from_numpy_matrix(network.todense())
functions[FORMAT](G, OP_FILE)

np.save(LABEL_FILE, labels.todense())
