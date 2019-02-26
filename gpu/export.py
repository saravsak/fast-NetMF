import pdb
import numpy as np

with open("mapping.txt",'r') as fp:
    lines = fp.readlines()
    mapping = {}
    for line in lines:
        source,target = line.strip().split(':')
        source = int(source)
        target = int(target)
        mapping[target] = source

embedding = []

with open("ppi.emb",'r') as fp:
    lines = fp.readlines()
    num_nodes, dim = lines[0].strip().split(' ')
    num_nodes = int(num_nodes)
    dim = int(dim)

    pdb.set_trace()
    embedding = np.zeros((num_nodes,dim))

    for line in lines[1:]:
        els = [float(el) for el in line.strip().split(' ')]
        embedding[mapping[int(els[0])]] = els[1:]

pdb.set_trace()
