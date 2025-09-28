import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx

# Input: edge list file
input_file = 'graph.txt'  # format: n m \n u1 v1 \n u2 v2 ...

# Step 1: Read edge list
with open(input_file, 'r') as f:
    n, m = map(int, f.readline().split())
    edges = []
    for line in f:
        u, v = map(int, line.strip().split())
        edges.append((u, v))

# Step 2: Build graph using networkx
G = nx.Graph()
G.add_nodes_from(range(1, n+1))  # assuming vertices numbered 1..n
G.add_edges_from(edges)

# Step 3: Compute features per vertex
vertex_features = []

# Optional: community detection (Louvain)
try:
    import community as community_louvain
    partition = community_louvain.best_partition(G)
except:
    # fallback if not installed
    partition = {v: 0 for v in G.nodes()}

for v in G.nodes():
    degree = G.degree(v)
    clustering_coeff = nx.clustering(G, v)
    # Approximate betweenness centrality (faster for large graphs)
    # Here we precompute for small graphs; for large graphs use approximate
    betweenness = nx.betweenness_centrality(G, k=min(100,n)).get(v, 0.0)
    community_id = partition[v]
    
    neighbors = list(G.neighbors(v))
    neighbor_degrees = [G.degree(u) for u in neighbors]
    
    if len(neighbor_degrees) > 0:
        avg_neighbor_degree = np.mean(neighbor_degrees)
        var_neighbor_degree = np.var(neighbor_degrees)
    else:
        avg_neighbor_degree = 0.0
        var_neighbor_degree = 0.0

    historical_frontier_hits = 0  # placeholder, can be added if past run exists
    
    memory_estimate = degree * 8  # assume 8 bytes per edge (adjust as needed)
    
    adjacency_size = degree  # same as degree
    
    avg_weight = 1.0  # unweighted, set 1.0; replace if weighted graph
    
    vertex_features.append({
        'vertex_id': v,
        'degree': degree,
        'clustering_coeff': clustering_coeff,
        'betweenness': betweenness,
        'community_id': community_id,
        'avg_neighbor_degree': avg_neighbor_degree,
        'var_neighbor_degree': var_neighbor_degree,
        'historical_frontier_hits': historical_frontier_hits,
        'memory_estimate': memory_estimate,
        'adjacency_size': adjacency_size,
        'avg_weight': avg_weight,
        'label': 0  # placeholder, to be filled by profiling
    })

# Step 4: Write CSV for ML
df = pd.DataFrame(vertex_features)
df.to_csv('vertex_features.csv', index=False)
print("CSV saved as vertex_features.csv")

