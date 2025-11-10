"""
inspect_graph.py

Quick utility to inspect a saved PyTorch Geometric graph (.pt)
produced by hexes_to_graph.py.

Usage:
    python3 inspect_graph.py --input graph_res8.pt
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect a saved PyTorch Geometric graph.")
    parser.add_argument("--input", "-i", required=True, help="Path to .pt graph file")
    args = parser.parse_args()

    # Load graph
    data = torch.load(args.input, map_location="cpu", weights_only=False)

    print("=== Graph Summary ===")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of node features: {data.num_node_features}")
    print(f"Feature columns (amenities + coords): {getattr(data, 'feature_columns', None)}")
    print(f"H3 resolution: {getattr(data, 'resolution', None)}\n")

    # -----------------------------
    # Node features
    # -----------------------------
    print("=== Example node features (first 5 nodes) ===")
    n_show = min(5, data.num_nodes)
    for i in range(n_show):
        h3_id = data.h3_ids[i] if hasattr(data, "h3_ids") else "<unknown>"
        feats = data.x[i].tolist()
        print(f"Node {i}: H3 = {h3_id}, x = {feats}")

    # -----------------------------
    # Edge list
    # -----------------------------
    print("\n=== Example edges (first 10 edges) ===")
    edge_index = data.edge_index
    n_edges = edge_index.shape[1]
    for i in range(min(10, n_edges)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        src_h3 = data.h3_ids[src] if hasattr(data, "h3_ids") else src
        dst_h3 = data.h3_ids[dst] if hasattr(data, "h3_ids") else dst
        print(f"Edge {i}: {src} → {dst}  (H3: {src_h3} → {dst_h3})")


if __name__ == "__main__":
    main()
