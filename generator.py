import networkx as nx
import random
from BayesNet import BayesNet
import itertools
import pandas as pd


class BNGenerator:
    def __init__(self, n_nodes, ew=0.5):
        self.n_nodes = n_nodes
        self.ew = ew
        self.bayes_net = BayesNet()
        G = nx.gnp_random_graph(n_nodes, self.ew, directed=True)
        self.bayes_net.structure = nx.DiGraph([(str(u), str(v), {'weight': random.randint(-10, 10)}) for (u, v) in G.edges() if u < v])
        for node in self.bayes_net.structure.nodes:
            edges = list(self.bayes_net.structure.edges(node))
            columns = [str(e) for _, e in edges] + ([node] if not len(edges) else [str(edges[0][0])]) + ['p']
            cpt = []
            rp = 1
            reversed_list = list(reversed(list(itertools.product([True, False], repeat=len(set(columns) - {'p'})))))
            for tv in reversed_list:
                rp = 1 if rp == 0 else rp
                p = random.random() if rp == 1 else rp
                rp = rp - p if rp == 1 else 0
                cpt.append(list(tv) + [round(p, 2)])
            cpts = pd.DataFrame(cpt, columns=columns)
            self.bayes_net.structure.add_node(node, cpt=cpts)
