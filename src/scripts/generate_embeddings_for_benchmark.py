import argparse
import torch
import pickle
import sys
import os
import numpy as np
import rootutils

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
    model = GraphMAE.load_from_checkpoint(args.checkpoint_path, weights_only=False)
    model.to(args.device)
    model.eval()

    # 3. Compute Embeddings
    print("Computing embeddings...")
    x = graph.x.to(args.device)
    edge_index = graph.edge_index.to(args.device)
    
    # Check input dimension mismatch
    try:
        expected_dim = None
        # Try to get expected dim from the first layer of the encoder
        if hasattr(model.model.encoder, 'gat_layers'):
            # GAT
            layer = model.model.encoder.gat_layers[0]
            if hasattr(layer, 'lin_src'):
                expected_dim = layer.lin_src.weight.shape[1]
        
        if expected_dim is None:
             # Fallback: try to infer from weight shape of first parameter that looks like input projection
             for name, param in model.model.encoder.named_parameters():
                 if 'weight' in name and param.shape[1] > 10: # heuristic
                     expected_dim = param.shape[1]
                     break
        
        if expected_dim is not None:
            current_dim = x.shape[1]
            if current_dim != expected_dim:
                print(f"Warning: Input feature dimension mismatch! Graph: {current_dim}, Model: {expected_dim}")
                if current_dim < expected_dim:
                    diff = expected_dim - current_dim
                    print(f"Padding input with {diff} zero columns...")
                    padding = torch.zeros((x.shape[0], diff), device=x.device)
                    x = torch.cat([x, padding], dim=1)
                else:
                    print(f"Truncating input to {expected_dim} columns...")
                    x = x[:, :expected_dim]
        else:
            print("Could not determine expected input dimension from model structure.")

    except Exception as e:
        print(f"Could not verify input dimensions (proceeding anyway): {e}")

    with torch.no_grad():
        # Access the internal model's embed method
        # GraphMAE wraps the model in self.model
        # The internal model (e.g. PreModel) has an embed method
        embeddings = model.model.embed(x, edge_index)
    
    embeddings_np = embeddings.cpu().numpy()
    
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
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Saved embeddings to {args.output_file}")

if __name__ == "__main__":
    main()
