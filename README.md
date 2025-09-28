Graph Processing and BFS with GPU/CPU Partitioning
This project provides a pipeline for graph processing, feature extraction, machine learning-based GPU/CPU partitioning, and BFS execution with hybrid device support.

Project Structure
text
HiPC_Poster_Code/
├── BFS_HiPC.cu                 # CUDA BFS implementation with device partitioning
├── Convert_To_CSR.cpp          # Convert graph to CSR format
├── Convert_To_CSV.py           # Convert assignment to partition format
├── Feature_Extract.py          # Serial feature extraction
├── Parallel_Feature_Extract.py # Parallel feature extraction
├── Parallel_Feature_Extract2.py # Enhanced parallel feature extraction
├── Hotness_List.py            # ML model for GPU/CPU assignment
├── Serial_BFS.cpp             # Serial BFS implementation
└── graph.txt                  # Sample input graph
Prerequisites
Python 3.x

CUDA Toolkit

GCC/G++ compiler

Python packages: pandas, scikit-learn, xgboost, joblib

Installation
Create and activate virtual environment:

bash
python3 -m venv venv
source venv/bin/activate
Install required Python packages:

bash
pip install pandas scikit-learn xgboost joblib
Execution Pipeline
Step 1: Feature Extraction
Extract graph features using parallel processing:

bash
python3 Parallel_Feature_Extract2.py <graph_file> <output_csv> <n_cores>
Example:

bash
python3 Parallel_Feature_Extract2.py graph.txt vertex_features.csv 250
Step 2: GPU/CPU Assignment
Train ML model and assign vertices to GPU/CPU:

bash
python3 Hotness_List.py <features_csv>
Example:

bash
python3 Hotness_List.py vertex_features.csv
Step 3: Convert to Partition Format
Convert assignments to partition file:

bash
python3 Convert_To_CSV.py <assignment_csv> <partition_csv>
Example:

bash
python3 Convert_To_CSV.py vertex_assignment.csv vertex_partition.csv
Step 4: Compile and Run BFS
Compile CUDA BFS and execute with partitioning:

bash
nvcc BFS_HiPC.cu -o bfs_executable
./bfs_executable <graph_file> <partition_file> <source_vertex>
Example:

bash
nvcc BFS_HiPC.cu -o bfs_executable
./bfs_executable graph.txt vertex_partition.csv 1
File Formats
Input Graph Format (graph.txt)
text
<num_vertices> <num_edges>
<source> <destination>
<source> <destination>
...
Feature File Format (vertex_features.csv)
text
vertex_id,degree,avg_neighbor_degree,var_neighbor_degree,memory_estimate,adjacency_size
0,1,4.0,0.0,8,1
1,4,2.25,1.1875,32,4
...
Partition File Format (vertex_partition.csv)
text
vertex_id,device
0,0
1,0
2,1
...
Where device=0 indicates CPU and device=1 indicates GPU.

BFS Output Format (bfs_output.csv)
text
vertex,level
0,1
1,0
2,1
...
Complete Example Run
bash
# 1. Extract features from graph
python3 Parallel_Feature_Extract2.py graph.txt vertex_features.csv 250

# 2. Generate GPU/CPU assignments using ML
python3 Hotness_List.py vertex_features.csv

# 3. Convert to partition format
python3 Convert_To_CSV.py vertex_assignment.csv vertex_partition.csv

# 4. Compile and run BFS with partitioning
nvcc BFS_HiPC.cu -o bfs_executable
./bfs_executable graph.txt vertex_partition.csv 1
Output Files Generated
vertex_features.csv: Graph features for each vertex

vertex_assignment.csv: ML predictions for GPU/CPU assignment

vertex_partition.csv: Partition file for BFS execution

bfs_output.csv: BFS traversal results with levels

Notes
The ML model uses XGBoost to predict optimal GPU/CPU assignments based on graph features

Features include degree, neighbor degree statistics, memory estimates, etc.

The system automatically handles cases with limited training data

BFS execution time is measured and displayed in milliseconds

Troubleshooting
Ensure CUDA is properly installed and nvcc is in PATH

Check that all Python dependencies are installed in the virtual environment

Verify input graph format matches expected structure

For large graphs, adjust the number of cores in feature extraction accordingly

Performance
Feature extraction scales with number of cores (up to 250 tested)

ML model provides intelligent partitioning for hybrid execution

BFS leverages both GPU and CPU based on the partitioning scheme


