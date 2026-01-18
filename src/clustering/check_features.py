import torch

# Load your graph
graph = torch.load("dataset_aligned/warsaw_hexagons_res9.pt", map_location="cpu", weights_only=False)

# 1. Print all available keys/attributes
print("Keys in graph:", graph.keys())

# 2. Check specifically for common name attributes
if hasattr(graph, 'feature_names'):
    print("\nFound feature names:", graph.feature_names)
elif hasattr(graph, 'col_names'):
    print("\nFound column names:", graph.col_names)
else:
    print("\nNo explicit feature names found. You must define them manually based on the paper.")
    
# 3. Check dimensions to verify count
print(f"\nNumber of features in graph: {graph.num_node_features}")