import networkx as nx

G = nx.complete_graph(4)

print(nx.shortest_path(G, 3, 0))
