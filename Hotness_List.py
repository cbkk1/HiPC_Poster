import sys
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================
# Step 1: Load CSV from CLI
# ==========================
if len(sys.argv) < 2:
    print("Usage: python Hotness_List.py <input_csv>")
    sys.exit(1)

input_file = sys.argv[1]
data = pd.read_csv(input_file)

# ==========================
# Step 2: Define heuristic labels if missing
# ==========================
# If 'label' column exists, use it; otherwise create heuristic
if 'label' not in data.columns:
    print("No label column found. Generating heuristic labels based on memory_estimate...")
    # Example heuristic: vertices with memory_estimate < threshold go to GPU
    gpu_memory_limit = data['memory_estimate'].quantile(0.3)  # smallest 30% of vertices
    data['label'] = (data['memory_estimate'] <= gpu_memory_limit).astype(int)

# ==========================
# Step 3: Feature selection
# ==========================
feature_cols = [
    'degree', 'clustering_coeff', 'betweenness', 'community_id',
    'avg_neighbor_degree', 'var_neighbor_degree', 'historical_frontier_hits',
    'memory_estimate', 'adjacency_size', 'avg_weight'
]

# Keep only features available in CSV
feature_cols = [c for c in feature_cols if c in data.columns]

X = data[feature_cols].copy()
y = data['label'].copy()

# ==========================
# Step 4: Encode categorical features
# ==========================
if 'community_id' in X.columns:
    X.loc[:, 'community_id'] = X['community_id'].astype('category').cat.codes

# ==========================
# Step 5: Train/Test split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# Step 6: XGBoost classifier
# ==========================
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='logloss',
    n_jobs=-1   # parallel training
)

model.fit(X_train, y_train)

# ==========================
# Step 7: Evaluate
# ==========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# ==========================
# Step 8: Predict GPU/CPU assignment for all vertices
# ==========================
data['gpu_assignment'] = model.predict(X)

# ==========================
# Step 9: Save to CSV
# ==========================
out_file = "vertex_assignment.csv"
data.to_csv(out_file, index=False)
print(f"Predictions saved to {out_file}")
