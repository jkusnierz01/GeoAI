import argparse
import torch
import pickle
import sys
import os
import numpy as np
import rootutils
from torch_geometric.utils import k_hop_subgraph
import h3

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
    
    # Print number of nodes in the graph file (before loading)
    try:
        import torch_geometric
        torch.serialization.add_safe_globals([
            torch_geometric.data.batch.DynamicInheritanceGetter
        ])
        graph_data = torch.load(args.graph_file, map_location='cpu', weights_only=False)
        if hasattr(graph_data, 'num_nodes'):
            print(f"[PRE-LOAD] Number of nodes in graph file: {graph_data.num_nodes}")
        elif isinstance(graph_data, dict) and 'num_nodes' in graph_data:
            print(f"[PRE-LOAD] Number of nodes in graph file: {graph_data['num_nodes']}")
        elif hasattr(graph_data, 'x'):
            print(f"[PRE-LOAD] Number of nodes in graph file: {graph_data.x.shape[0]}")
        else:
            print("[PRE-LOAD] Could not determine number of nodes in graph file.")
    except Exception as e:
        print(f"[PRE-LOAD] Could not load graph file to count nodes: {e}")
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

    # Debug: print h3_ids length and sample
    h3_ids_debug = None
    if hasattr(graph, 'h3_ids'):
        h3_ids_debug = graph.h3_ids
    elif hasattr(graph, 'node_names'):
        h3_ids_debug = graph.node_names
    # Flatten if list of lists
    if h3_ids_debug is not None and len(h3_ids_debug) > 0 and isinstance(h3_ids_debug[0], list):
        print("h3_ids is a list of lists, flattening...")
        h3_ids_debug = [item for sublist in h3_ids_debug for item in sublist]
        if hasattr(graph, 'h3_ids'):
            graph.h3_ids = h3_ids_debug
        elif hasattr(graph, 'node_names'):
            graph.node_names = h3_ids_debug
    if h3_ids_debug is not None:
        print(f"Loaded h3_ids length: {len(h3_ids_debug)}")
        print(f"First 10 h3_ids: {h3_ids_debug[:10]}")
    else:
        print("No h3_ids or node_names attribute found in graph.")

    # --- Identify isolated nodes (nodes with no edges) ---
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    device = edge_index.device
    degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long, device=device))
    degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long, device=device))
    isolated_nodes = (degree == 0).nonzero(as_tuple=True)[0]
    print(f"Found {isolated_nodes.numel()} isolated nodes.")
    # Find index of 'is_empty' feature
    x = graph.x
    num_features = x.shape[1]
    is_empty_idx = None
    for i, name in enumerate(getattr(graph, 'feature_names', [])):
        if name == 'is_empty':
            is_empty_idx = i
            break
    if is_empty_idx is None:
        if hasattr(graph, 'feature_names') and len(graph.feature_names) == num_features:
            is_empty_idx = num_features - 1
        else:
            is_empty_idx = num_features - 1

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
    valid_h3_indices = []
    h3_ids = None
    if hasattr(graph, 'h3_ids'):
        h3_ids = graph.h3_ids
    elif hasattr(graph, 'node_names'):
        h3_ids = graph.node_names
    else:
        h3_ids = None

    with torch.no_grad():
        for node_idx in range(graph.num_nodes):
            # Only process nodes with a valid H3 ID
            if h3_ids is not None and node_idx >= len(h3_ids):
                continue
            if node_idx in isolated_nodes:
                # --- TEMPORARY HEXAGON SUBGRAPH FOR ISOLATED NODE ---
                # Get H3 ID of the isolated node
                if h3_ids is not None:
                    center_h3 = h3_ids[node_idx]
                else:
                    center_h3 = None
                # Get 6 neighbor H3 IDs using h3.grid_disk (returns center + neighbors)
                if center_h3 is not None:
                    neighbors = list(h3.grid_disk(center_h3, 1))
                    if center_h3 in neighbors:
                        neighbors.remove(center_h3)
                    # Only keep 6 neighbors (should be exactly 6 for valid H3 index)
                    neighbors = neighbors[:6]
                else:
                    neighbors = [f'virtual_{node_idx}_{i}' for i in range(6)]
                # Build features: center node + 6 virtual neighbors
                x_center = x_full[node_idx].unsqueeze(0)
                x_virtual = torch.zeros((6, num_features), device=x_full.device)
                x_virtual[:, is_empty_idx] = 1
                x_sub = torch.cat([x_center, x_virtual], dim=0)
                # Build edge_index: connect center to each, and hexagon ring
                edges = []
                for i in range(6):
                    edges.append([0, i+1])  # center to neighbor
                    edges.append([i+1, 0])  # neighbor to center
                for i in range(6):
                    n1 = i+1
                    n2 = ((i+1) % 6) + 1
                    edges.append([n1, n2])
                    edges.append([n2, n1])
                sub_edge_index = torch.tensor(edges, dtype=edge_index.dtype, device=x_full.device).t()
                mapping = 0  # center node is always index 0
                sub_embeds = model.model.embed(x_sub, sub_edge_index)
                center_embed = sub_embeds[mapping].cpu().numpy()
                embeddings_np.append(center_embed)
                valid_h3_indices.append(node_idx)
            else:
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
                valid_h3_indices.append(node_idx)
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

    if len(valid_h3_indices) != len(embeddings_np):
        print(f"Error: Number of valid H3 indices ({len(valid_h3_indices)}) does not match number of embeddings ({len(embeddings_np)})")
        return

    result = {}
    for i, node_idx in enumerate(valid_h3_indices):
        h3_id = h3_ids[node_idx]
        # Ensure h3_id is hashable (convert list to string if needed)
        if isinstance(h3_id, list):
            h3_id = ','.join(map(str, h3_id))
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
