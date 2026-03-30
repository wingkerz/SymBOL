import pandas as pd
import torch
import networkx as nx
import numpy as np
from scipy.sparse import *
import torch
# from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
from sympy import false


def grid_8_neighbor_graph(N):
    """
    Build discrete grid graph, each node has 8 neighbors
    :param n:  sqrt of the number of nodes
    :return:  A, the adjacency matrix
    """
    N = int(N)
    n = int(N ** 2)
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    A = torch.zeros(n, n)
    for x in range(N):
        for y in range(N):
            index = x * N + y
            for i in range(len(dx)):
                newx = x + dx[i]
                newy = y + dy[i]
                if N > newx >= 0 and N > newy >= 0:
                    index2 = newx * N + newy
                    A[index, index2] = 1
    return A.float()


class Topo:
    def __init__(self, N, topo_type):
        self.N = N
        self.topo_type = topo_type
        self.G = self.build_topology(N, topo_type)
        self.sparse_adj = self.trans_into_sparse_adj(self.G)
        self.adj = self.get_dense_adj() 

    def update_N(self, N):
        self.N = N
    def build_topology(self, N, topo_type, **params):
        """
        :param N: #nodes
        :param topo_type: the type of topology
        :param params:
        :return: G
        """
        print("building network topology [%s] ..." % topo_type)
        if topo_type == 'grid':
            nn = int(np.ceil(np.sqrt(N)))  # grid-layout pixels :20
            A = grid_8_neighbor_graph(nn)
            G = nx.from_numpy_array(A.numpy())
            self.update_N(int(nn*nn))
        elif topo_type == 'random':
            if 'p' in params:
                p = params['p']
                print("setting p to %s ..." % p)
            else:
                print("setting default values [0.1] to p ...")
                p = 0.1
            G = nx.erdos_renyi_graph(N, p)
            # print(G)
        elif topo_type == 'power_law':
            if 'm' in params:
                m = params['m']
                print("setting m to %s ..." % m)
            else:
                print("setting default values [5] to m ...")
                m = 2
            if N <= m:
                N = N + m
            G = nx.barabasi_albert_graph(N, m)
            # print(G)
        elif topo_type == 'small_world':
            if 'k' in params:
                k = params['k']
                print("setting k to %s ..." % k)
            else:
                print("setting default values [5] to k ...")
                k = 5
            if 'p' in params:
                p = params['p']
                print("setting p to %s ..." % p)
            else:
                print("setting default values [0.5] to p ...")
                p = 0.5
            G = nx.newman_watts_strogatz_graph(N, k, p)
        elif topo_type == 'community':
            n1 = int(N / 3)
            n2 = int(N / 3)
            n3 = int(N / 4)
            n4 = N - n1 - n2 - n3

            if 'p_in' in params:
                p_in = params['p_in']
                print("setting p_in to %s ..." % p_in)
            else:
                print("setting default values [0.25] to p_in ...")
                p_in = 0.25
            if 'p_out' in params:
                p_out = params['p_out']
                print("setting p_out to %s ..." % p_out)
            else:
                print("setting default values [0.01] to p_out ...")
                p_out = 0.01
            G = nx.random_partition_graph([n1, n2, n3, n4], p_in, p_out)
        elif topo_type == 'full_connected':
            G = nx.complete_graph(N)
            # G.add_edges_from([(i, i) for i in range(N)])  # add self_loop
        elif topo_type == 'directed_full_connected':
            G = nx.complete_graph(N, nx.DiGraph())
            # G.add_edges_from([(i, i) for i in range(N)])  # add self_loop
        else:
            print("ERROR topo_type [%s]" % topo_type)
            exit(1)
        return G

    def trans_into_sparse_adj(self, G):
        # A = np.array(nx.adjacency_matrix(G).todense(), dtype=np.float64)
        #sparse_A = coo_matrix(A)
        sparse_A = coo_matrix(nx.adjacency_matrix(G))
        row, col = torch.from_numpy(sparse_A.row).long(), torch.from_numpy(sparse_A.col).long()
        return torch.cat([row.view(1, -1), col.view(1, -1)], dim=0)

    def get_dense_adj(self):
        A = nx.adjacency_matrix(self.G).todense()
        return torch.tensor(A, dtype=torch.float32)

if __name__ == "__main__":
    N = 100
    topo_type = 'power_law'
    topo = Topo(N, topo_type)
