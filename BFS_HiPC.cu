#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <omp.h>

#define THREADS 256

// ================= GPU Kernel =================
__global__ void bfs_gpu_kernel(
    int *d_row_ptr,
    int *d_col_idx,
    int *d_frontier,
    int *d_next_frontier,
    int *d_visited,
    int *d_level,
    int n,
    int depth
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    if (!d_frontier[u]) return;

    for (int e = d_row_ptr[u]; e < d_row_ptr[u+1]; e++) {
        int v = d_col_idx[e];
        if (atomicCAS(&d_visited[v], 0, 1) == 0) {
            d_level[v] = depth + 1;
            d_next_frontier[v] = 1;
        }
    }
}

// ================= CSR Builder =================
void buildCSR(int n, int m, int (*edges)[2], int *row_ptr, int *col_idx) {
    int *deg = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = edges[i][0], v = edges[i][1];
        deg[u]++;
        deg[v]++; // undirected
    }

    row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        row_ptr[i+1] = row_ptr[i] + deg[i];
    }

    int *temp = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) temp[i] = row_ptr[i];

    for (int i = 0; i < m; i++) {
        int u = edges[i][0], v = edges[i][1];
        col_idx[temp[u]++] = v;
        col_idx[temp[v]++] = u;
    }

    free(temp);
    free(deg);
}

// ================= Main =================
int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <graph.txt> <partition.csv> <source>\n", argv[0]);
        return 1;
    }

    char *graphFile = argv[1];
    char *partFile  = argv[2];
    int source      = atoi(argv[3]);

    // ---- Read graph ----
    FILE *fin = fopen(graphFile, "r");
    if (!fin) { printf("Error opening %s\n", graphFile); return 1; }

    int n, m;
    fscanf(fin, "%d %d", &n, &m);
    int (*edges)[2] = (int(*)[2]) malloc(m * sizeof(int[2]));
    for (int i = 0; i < m; i++) fscanf(fin, "%d %d", &edges[i][0], &edges[i][1]);
    fclose(fin);

    // ---- Read partition ----
    int *isGPU = (int*)calloc(n, sizeof(int));
    int *isCPU = (int*)calloc(n, sizeof(int));

    FILE *pin = fopen(partFile, "r");
    if (!pin) { printf("Error opening %s\n", partFile); return 1; }
    char line[128];
    fgets(line, sizeof(line), pin); // skip header
    while (fgets(line, sizeof(line), pin)) {
        int vid;
        char part[16];
        sscanf(line, "%d,%15s", &vid, part);
        if (strcmp(part, "GPU") == 0) isGPU[vid] = 1;
        else isCPU[vid] = 1;
    }
    fclose(pin);

    // ---- Build CSR ----
    int *row_ptr = (int*)malloc((n+1)*sizeof(int));
    int *col_idx = (int*)malloc(2*m*sizeof(int));
    buildCSR(n, m, edges, row_ptr, col_idx);
    free(edges);

    // ---- Host arrays ----
    int *frontier      = (int*)calloc(n, sizeof(int));
    int *next_frontier = (int*)calloc(n, sizeof(int));
    int *visited       = (int*)calloc(n, sizeof(int));
    int *level         = (int*)malloc(n*sizeof(int));
    for (int i = 0; i < n; i++) level[i] = -1;

    frontier[source] = 1;
    visited[source] = 1;
    level[source]   = 0;

    // ---- Device arrays ----
    int *d_row_ptr, *d_col_idx, *d_frontier, *d_next_frontier, *d_visited, *d_level;
    cudaMalloc(&d_row_ptr, (n+1)*sizeof(int));
    cudaMalloc(&d_col_idx, (2*m)*sizeof(int));
    cudaMalloc(&d_frontier, n*sizeof(int));
    cudaMalloc(&d_next_frontier, n*sizeof(int));
    cudaMalloc(&d_visited, n*sizeof(int));
    cudaMalloc(&d_level, n*sizeof(int));

    cudaMemcpy(d_row_ptr, row_ptr, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, (2*m)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, level, n*sizeof(int), cudaMemcpyHostToDevice);

    // ---- BFS Loop ----

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int depth = 0;
    while (1) {
        memset(next_frontier, 0, n*sizeof(int));
        cudaMemset(d_next_frontier, 0, n*sizeof(int));

        cudaMemcpy(d_frontier, frontier, n*sizeof(int), cudaMemcpyHostToDevice);

        // GPU phase
        int blocks = (n + THREADS - 1) / THREADS;
        bfs_gpu_kernel<<<blocks, THREADS>>>(d_row_ptr, d_col_idx,
                                            d_frontier, d_next_frontier,
                                            d_visited, d_level, n, depth);


        // CPU phase
        #pragma omp parallel for schedule(dynamic,64)
        for (int u = 0; u < n; u++) {
            if (!isCPU[u] || !frontier[u]) continue;
            for (int e = row_ptr[u]; e < row_ptr[u+1]; e++) {
                int v = col_idx[e];
                if (!visited[v]) {


                            visited[v] = 1;
                            level[v] = depth+1;
                            next_frontier[v] = 1;
                }
            }
        }
                cudaDeviceSynchronize();

        // Merge GPU results
        int *gpu_next = (int*)malloc(n*sizeof(int));
        cudaMemcpy(gpu_next, d_next_frontier, n*sizeof(int), cudaMemcpyDeviceToHost);

        int any_set = 0;


        for (int i = 0; i < n; i++) {
            frontier[i] = next_frontier[i] || gpu_next[i];
            if (frontier[i]) { visited[i] = 1; any_set = 1; }
        }
        free(gpu_next);

        if (!any_set) break;
        depth++;
    }

        cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("BFS Time: %f ms\n", milliseconds);  

    // ---- Save BFS output ----
    FILE *fout = fopen("bfs_output.csv", "w");
    fprintf(fout, "vertex,level\n");
    for (int i = 0; i < n; i++) {
        fprintf(fout, "%d,%d\n", i, level[i]);
    }
    fclose(fout);

    printf("BFS completed. Results saved to bfs_output.csv\n");

    // ---- Cleanup ----
    free(row_ptr); free(col_idx);
    free(frontier); free(next_frontier);
    free(visited); free(level);
    free(isGPU); free(isCPU);
    cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_frontier); cudaFree(d_next_frontier);
    cudaFree(d_visited); cudaFree(d_level);

    return 0;
}
