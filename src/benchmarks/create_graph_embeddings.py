import csv
import argparse
import numpy as np
import torch
import h3
from tqdm import tqdm

# Imports from your project structure
from utils.graph_utils import prepare_graph
from utils.model_utils import load_model_from_checkpoint, get_k_hop_subgraph_embedding

def read_h3_ids(csv_path):
    """Reads H3 IDs from the first column of a CSV file."""
    h3_ids = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                h3_ids.append(row[0])
    return h3_ids

def get_node_index_for_h3(graph, target_h3_id):
    """
    Finds the integer node index in the graph for a given H3 ID string.
    
    IMPORTANT: This assumes your graph object has an attribute 'h3_ids' 
    or similar metadata that maps nodes to H3 strings.
    """
    if hasattr(graph, 'h3_ids'):
        try:
            return graph.h3_ids.index(target_h3_id)
        except ValueError:
            return None
    
    if hasattr(graph, 'node_names'):
         try:
            return graph.node_names.index(target_h3_id)
         except ValueError:
            return None
            
    return None

def get_center_child_h3_id(h3_id, new_resolution):
    res = h3.get_resolution(h3_id)
    center_child = h3.cell_to_center_child(h3_id, new_resolution)
    return center_child

def main():
    parser = argparse.ArgumentParser(description="Calculate GraphMAE embeddings for H3 IDs from CSV")
    
    # Input files
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV containing H3 IDs")
    parser.add_argument("--graph_file", type=str, required=True, help="Path to the graph file (e.g., .pt)")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config_path", type=str, default="configs/defaults.yaml")
    
    # Embedding parameters
    parser.add_argument("--k_hop", type=int, default=1, help="K-hop size for the subgraph embedding")
    parser.add_argument("--output_file", type=str, default="h3_embeddings.npy", help="Where to save the result")
    
    args = parser.parse_args()

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Read H3 IDs from CSV
    print(f"Reading H3 IDs from {args.csv_file}...")
    target_h3_ids = read_h3_ids(args.csv_file)
    # Optional: Skip header if the first row is not a valid H3 ID
    print(f"Skipping header: {target_h3_ids[0]}")
    res = h3.get_resolution(target_h3_ids[1])
    target_h3_ids = [get_center_child_h3_id(h3_id, res + 1) for h3_id in target_h3_ids[1:]]
    
    print(f"Found {len(target_h3_ids)} H3 IDs to process.")

    # 3. Load Graph
    print(f"Loading graph from {args.graph_file}...")
    graph = prepare_graph(args.graph_file) 
    

    print(f"Graph loaded. Nodes: {graph.num_nodes}, Features: {graph.num_node_features}")

    # 4. Load Model
    try:
        num_classes = max(graph.y.max().item() for g in [graph]) + 1
    except:
        num_classes = 0
    model = load_model_from_checkpoint(
        args.model_path, 
        graph.num_node_features, 
        num_classes,
        config_path=args.config_path, 
        device=device
    )
    model.eval()

    embeddings = {}
    
    print(f"Calculating embeddings with k_hop={args.k_hop}...")
    for h3_id in tqdm(target_h3_ids):
        node_idx = get_node_index_for_h3(graph, h3_id)
        
        if node_idx is None:
            continue
            
        try:
            emb = get_k_hop_subgraph_embedding(graph, model, node_idx, args.k_hop, device)
            
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
                
            embeddings[h3_id] = emb
            
        except Exception as e:
            print(f"Error processing {h3_id} (Node {node_idx}): {e}")

    # 6. Save Results
    print(f"Successfully computed {len(embeddings)} embeddings.")
    np.save(args.output_file, embeddings)
    print(f"Saved dictionary to {args.output_file}")

if __name__ == "__main__":
    main()