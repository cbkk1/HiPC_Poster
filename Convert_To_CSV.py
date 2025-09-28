import sys
import pandas as pd

if len(sys.argv) < 3:
    print("Usage: python convert_to_partition.py <input_features.csv> <output_partition.csv>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Load the big feature CSV
data = pd.read_csv(input_file)

# We only care about vertex_id and gpu_assignment
if "vertex_id" not in data.columns or "gpu_assignment" not in data.columns:
    print("Error: Input file must contain 'vertex_id' and 'gpu_assignment' columns")
    sys.exit(1)

partition = data[["vertex_id", "gpu_assignment"]].copy()

# Optional: rename gpu_assignment → device for clarity (0=CPU, 1=GPU)
partition.rename(columns={"gpu_assignment": "device"}, inplace=True)

# Save
partition.to_csv(output_file, index=False)
print(f"Partition file written to {output_file}")
