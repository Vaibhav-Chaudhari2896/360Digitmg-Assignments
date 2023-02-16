import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# Degree Centrality
G = pd.read_csv("C:\\Datasets_BA\\360DigiTMG\\DS_India\\360DigiTMG DS India Module wise PPTs\\Module 25 Network Analytics\\Dataset\\routes.csv")
G = G.iloc[:, 1:10]

g = nx.Graph()

g = nx.from_pandas_edgelist(G, source = 'Source Airport', target = 'Destination Airport')

print(nx.info(g))

b = nx.degree_centrality(g)  # Degree Centrality
print(b) 

pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')

# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)

## Betweeness Centrality 
b = nx.betweenness_centrality(g) # Betweeness_Centrality
print(b)

## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
print(evg)

# cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)

# Average clustering
cc = nx.average_clustering(g) 
print(cc)
