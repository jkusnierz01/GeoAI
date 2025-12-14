import argparse
import torch
import pickle
import sys
import os
import numpy as np
import rootutils
from torch_geometric.utils import k_hop_subgraph

# Setup root to allow imports from src
ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.models.graphmae_module import GraphMAE
from src.utils.graph_utils import prepare_graph

def main():
    parser = argparse.ArgumentParser(description="Generate GraphMAE embeddings for benchmark")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to Lightning checkpoint")
    parser.add_argument("--graph_file", type=str, required=True, help="Path to the graph file (.pt)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save pickle embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k_hop", type=int, default=1, help="K-hop size for the subgraph embedding")

    args = parser.parse_args()
    
    print(f"Using device: {args.device}")

    # 1. Load Graph
    print(f"Loading graph from {args.graph_file}...")
    try:
        # Use prepare_graph to handle edge_index slicing etc.
        graph = prepare_graph(args.graph_file)
    except Exception as e:
        print(f"Failed to load graph: {e}")
        return

    print(f"Graph loaded. Nodes: {graph.num_nodes}, Features: {graph.num_node_features}")

    # 2. Load Model
    print(f"Loading model from {args.checkpoint_path}...")
    # We load the LightningModule
    # Fix for PyTorch 2.6+ checkpoint unpickling errors with numpy objects
    import numpy as np
    torch.serialization.add_safe_globals([
        np.dtype,
        __import__('numpy')._core.multiarray.scalar,
        __import__('numpy').dtypes.Float64DType
    ])
    model = GraphMAE.load_from_checkpoint(args.checkpoint_path, weights_only=False)
    model.to(args.device)
    model.eval()

    # 3. Compute Embeddings

    print("Computing embeddings for each node with its k-hop neighborhood...")
    x_full = graph.x.to(args.device)
    edge_index_full = graph.edge_index.to(args.device)

    # Check input dimension mismatch for the full graph (to know expected_dim)
    try:
        expected_dim = None
        if hasattr(model.model.encoder, 'gat_layers'):
            layer = model.model.encoder.gat_layers[0]
            if hasattr(layer, 'lin_src'):
                expected_dim = layer.lin_src.weight.shape[1]
        if expected_dim is None:
            for name, param in model.model.encoder.named_parameters():
                if 'weight' in name and param.shape[1] > 10:
                    expected_dim = param.shape[1]
                    break
    except Exception as e:
        print(f"Could not determine expected input dimension: {e}")

    embeddings_np = []
    with torch.no_grad():
        for node_idx in range(graph.num_nodes):
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                node_idx, args.k_hop, edge_index_full, relabel_nodes=True
            )
            x_sub = x_full[subset]
            # Pad/truncate if needed
            if expected_dim is not None:
                current_dim = x_sub.shape[1]
                if current_dim < expected_dim:
                    diff = expected_dim - current_dim
                    padding = torch.zeros((x_sub.shape[0], diff), device=x_sub.device)
                    x_sub = torch.cat([x_sub, padding], dim=1)
                elif current_dim > expected_dim:
                    x_sub = x_sub[:, :expected_dim]
            sub_edge_index = sub_edge_index.to(args.device)
            sub_embeds = model.model.embed(x_sub, sub_edge_index)
            center_embed = sub_embeds[mapping].cpu().numpy()
            embeddings_np.append(center_embed)
    embeddings_np = np.vstack(embeddings_np)
    
    # 4. Map to H3 IDs
    h3_ids = None
    if hasattr(graph, 'h3_ids'):
        h3_ids = graph.h3_ids
    elif hasattr(graph, 'node_names'):
        h3_ids = graph.node_names
    
    if h3_ids is None:
        print("Error: No 'h3_ids' or 'node_names' attribute found in the graph object. Cannot map embeddings to H3 IDs.")
        return

    if len(h3_ids) != len(embeddings_np):
        print(f"Error: Number of H3 IDs ({len(h3_ids)}) does not match number of embeddings ({len(embeddings_np)})")
        return
    
    result = {}
    for i, h3_id in enumerate(h3_ids):
        result[h3_id] = embeddings_np[i]
    
    print(f"Mapped {len(result)} embeddings to H3 IDs.")

    # 5. Save
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Saved embeddings to {args.output_file}")

if __name__ == "__main__":
    main()
