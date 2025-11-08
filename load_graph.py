import torch
data = torch.load("graph_res8.pt", weights_only=False)

print("=== Graph Summary ===")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of node features: {data.num_node_features}")
print(f"Feature columns (amenities + coords): {getattr(data,'feature_columns',None)}")
print(f"H3 resolution: {getattr(data,'resolution', None)}\n")

# -----------------------------
# Node features
# -----------------------------
print("=== Example node features (first 5 nodes) ===")
for i in range(min(5, data.num_nodes)):
    print(f"Node {i}: x = {data.x[i].tolist()}, H3 = {data.h3_ids[i]}")

# -----------------------------
# Edge list
# -----------------------------
print("\n=== Example edges (first 10 edges) ===")
# edge_index is shape [2, num_edges]
edge_index = data.edge_index
for i in range(min(10, edge_index.shape[1])):
    src = edge_index[0, i].item()
    dst = edge_index[1, i].item()
    print(f"Edge {i}: {src} -> {dst} (H3: {data.h3_ids[src]} -> {data.h3_ids[dst]})")
