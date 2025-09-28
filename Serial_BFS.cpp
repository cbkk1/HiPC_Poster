#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <chrono>
#include <climits>
#include <string>

void read_csr_file(const std::string& filename, 
                  std::vector<long long>& offsets, 
                  std::vector<long long>& indices,
                  long long& n, long long& m) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Read header (n = nodes, m = edges)
    infile >> n >> m;

    // Read offsets (vertex array) - n+1 elements
    offsets.resize(n + 1);
    for (long long i = 0; i <= n; i++) {
        infile >> offsets[i];
    }

    // Read indices (edge array) - m elements
    indices.resize(m);
    for (long long i = 0; i < m; i++) {
        infile >> indices[i];
    }

    infile.close();
}

void bfs_serial(long long source_vertex, long long num_vertices, 
               const long long* offsets, const long long* indices, 
               long long* level_array) {
    std::fill(level_array, level_array + num_vertices, INT_MAX);
    level_array[source_vertex] = 0;
    std::queue<long long> q;
    q.push(source_vertex);

    while (!q.empty()) {
        long long v = q.front();
        q.pop();
        long long curr_level = level_array[v];
        long long start = offsets[v];
        long long end = offsets[v + 1];
        for (long long i = start; i < end; ++i) {
            long long neighbor = indices[i];
            if (level_array[neighbor] == INT_MAX) {
                level_array[neighbor] = curr_level + 1;
                q.push(neighbor);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <CSR filename>\n";
        return 1;
    }
    
    std::string filename = argv[1];
    std::vector<long long> offsets, indices;
    long long n, m;
    
    read_csr_file(filename, offsets, indices, n, m);
    
    std::cout << "Nodes: " << n << "\n";
    std::cout << "Edges: " << m << "\n";
    
    // Print CSR for first 5 nodes
    /*for (long long i = 0; i < 5 && i < n; i++) {
        std::cout << "Node " << i << ": ";
        long long start = offsets[i];
        long long end = offsets[i + 1];
        for (long long j = start; j < end; j++) {
            std::cout << indices[j] << " ";
        }
        std::cout << "\n";
    }*/
    
    std::cout << "Enter source vertex (0-" << n-1 << "): ";
    long long source;
    std::cin >> source;
    
    if (source < 0 || source >= n) {
        std::cerr << "Invalid source vertex!" << std::endl;
        return 1;
    }
    
    std::vector<long long> levels(n);
    auto start_time = std::chrono::high_resolution_clock::now();
    bfs_serial(source, n, offsets.data(), indices.data(), levels.data());
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Write BFS results to file
    std::ofstream out("bfs_output.txt");
    for (long long i = 0; i < n; ++i) {
        if (levels[i] == INT_MAX) {
            out << i << " -1\n";
        } else {
            out << i << "," << levels[i] << "\n";
        }
    }
    out << "BFS execution time: " << elapsed.count() << " seconds\n";
    out.close();

    std::cout << "\nBFS results written to bfs_output.txt\n";
    std::cout << "BFS time: " << elapsed.count() << " seconds\n";

    return 0;
}