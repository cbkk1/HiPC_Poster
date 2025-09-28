# Parallel_Feature_Extract_Batch.py
import sys
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from math import ceil
from collections import defaultdict

# ----------------------------
# Process a batch of vertices
# ----------------------------
def process_vertex_batch(batch, adj, degrees):
    results = []
    for u in batch:
        neighbors = adj[u]
        deg = degrees[u]
        if deg > 0:
            avg_deg = np.mean([degrees[v] for v in neighbors])
            var_deg = np.var([degrees[v] for v in neighbors])
        else:
            avg_deg = 0.0
            var_deg = 0.0
        mem_estimate = deg * 8  # assuming 8 bytes per edge
        adjacency_size = deg
        results.append({
            "vertex_id": u,
            "degree": deg,
            "avg_neighbor_degree": avg_deg,
            "var_neighbor_degree": var_deg,
            "memory_estimate": mem_estimate,
            "adjacency_size": adjacency_size
        })
    return results

# ----------------------------
# Main
# ----------------------------
if len(sys.argv) < 4:
    print("Usage: python Parallel_Feature_Extract_Batch.py <graph_file> <output_csv> <n_cores>")
    sys.exit(1)

graph_file = sys.argv[1]
out_csv = sys.argv[2]
n_cores = int(sys.argv[3])

# ----------------------------
# Read graph
# ----------------------------
print("Reading graph...")
with open(graph_file, "r") as f:
    n, m = map(int, f.readline().split())
    adj = defaultdict(list)
    degrees = [0] * n
    for line in f:
        u, v = map(int, line.strip().split())
        adj[u].append(v)
        degrees[u] += 1

vertices = list(range(n))
batch_size = 1000  # batch size of vertices
num_batches = ceil(n / batch_size)
batches = [vertices[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]

# ----------------------------
# Parallel computation
# ----------------------------
print(f"Computing features with {n_cores} cores and {num_batches} batches...")
results = Parallel(n_jobs=n_cores, verbose=10)(
    delayed(process_vertex_batch)(batch, adj, degrees) for batch in batches
)

# Flatten results
flat_results = [item for batch_res in results for item in batch_res]

# ----------------------------
# Save CSV
# ----------------------------
df = pd.DataFrame(flat_results)
df.to_csv(out_csv, index=False)
print(f"Features saved to {out_csv}")

