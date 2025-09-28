# feature_extract_parallel.py
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def read_edge_list(filename):
    with open(filename) as f:
        n, m = map(int, f.readline().split())
        adj = [[] for _ in range(n)]
        for line in f:
            u, v = map(int, line.split())
            adj[u].append(v)
            adj[v].append(u)  # assume undirected
    return adj

def process_vertex(u, adj, degrees):
    deg_u = degrees[u]
    if deg_u > 0:
        neigh_deg = [degrees[v] for v in adj[u]]
        avg = float(np.mean(neigh_deg))
        var = float(np.var(neigh_deg))
    else:
        avg, var = 0.0, 0.0
    mem_est = deg_u * 8
    return (u, deg_u, avg, var, mem_est, deg_u)

def compute_features(adj, n_jobs=-1):
    n = len(adj)
    degrees = np.array([len(neigh) for neigh in adj], dtype=np.int32)

    results = Parallel(n_jobs=n_jobs, verbose=5, batch_size=1000)(
        delayed(process_vertex)(u, adj, degrees) for u in range(n)
    )

    df = pd.DataFrame(results, columns=[
        "vertex_id", "degree", "avg_neighbor_degree",
        "var_neighbor_degree", "memory_estimate", "adjacency_size"
    ])
    return df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python feature_extract_parallel.py graph.txt vertex_features.csv [n_jobs]")
        sys.exit(1)

    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else -1  # -1 = all cores

    print("Reading graph...")
    adj = read_edge_list(graph_file)

    print(f"Computing features with {n_jobs} cores...")
    df = compute_features(adj, n_jobs=n_jobs)

    print(f"Writing features to {out_file}")
    df.to_csv(out_file, index=False)

