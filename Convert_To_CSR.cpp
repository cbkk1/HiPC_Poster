#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];
    std::ifstream file(input_filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << input_filename << std::endl;
        return 1;
    }

    long long  n, m;
    file >> n >> m;

    std::vector<std::vector<long long >> adj(n);

    for (long long  i = 0; i < m; i++) {
        long long  u, v;
        file >> u >> v;
        adj[u].push_back(v);
    }
    file.close();

    // Build CSR arrays
    std::vector<long long > row_ptr(n + 1);
    std::vector<long long > col_idx;

    row_ptr[0] = 0;
    for (long long  i = 0; i < n; i++) {
        std::sort(adj[i].begin(), adj[i].end());
        row_ptr[i + 1] = row_ptr[i] + adj[i].size();
        col_idx.insert(col_idx.end(), adj[i].begin(), adj[i].end());
    }

    // Create output filename
    std::string output_filename = "CSR.txt";
    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        std::cerr << "Error creating file: " << output_filename << std::endl;
        return 1;
    }

    // Write to output file
    outfile << n << " " << m << "\n";

    for (long long  i = 0; i <= n; i++) {
        outfile << row_ptr[i] << " ";
    }
    outfile << "\n";

    for (long long  idx : col_idx) {
        outfile << idx << " ";
    }
    outfile << "\n";

    outfile.close();
    return 0;
}
