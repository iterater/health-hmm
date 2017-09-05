import networkx as nx
import numpy as np

G = nx.Graph()
G.add_nodes_from([1, 2, 3])
G.add_edges_from([(1,2), (2,3), (1,3)])
nx.draw(G)
