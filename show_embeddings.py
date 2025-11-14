import argparse
import numpy as np
from tqdm import tqdm
import torch
import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse # Make sure argparse is imported at the top
from types import SimpleNamespace # Import this at the top

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

# --- Import from GraphMAE ---
# Make sure you run this from the GeoAI directory
# or that GraphMAE is in your PYTHONPATH
from graphmae.utils import build_args
from graphmae.models import build_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def load_graph_data(graph_path):
    """
    Loads the original .pt graph, fixes edge_index,
    and adds dummy labels/masks.
    """
    print(f"--- Loading graph: {graph_path} ---")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    # Add dummy labels if they don't exist or are None
    if not hasattr(graph, 'y') or graph.y is None:
        graph.y = torch.zeros(graph.num_nodes, dtype=torch.long)

    # Fix edge_index shape
    if graph.edge_index.shape[0] == 4:
        graph.edge_index = graph.edge_index[:2, :]
    elif graph.edge_index.shape[0] != 2:
        raise ValueError("Unsupported edge_index shape")

    print(f"--- Graph loaded: {graph.num_nodes} nodes ---")
    return graph

def load_trained_model(checkpoint_path, num_features, num_classes):
    """
    Loads the pre-trained GraphMAE model from a checkpoint.
    """
    print(f"--- Loading model from: {checkpoint_path} ---")

    # 1. Create a "dummy" args object with the SAME DEFAULTS as training
    #    This avoids the build_args() conflict.
def load_trained_model(checkpoint_path, num_features, num_classes):
    """
    Loads the pre-trained GraphMAE model from a checkpoint.
    """
    print(f"--- Loading model from: {checkpoint_path} ---")

    # 1. Create a "dummy" args object with the SAME DEFAULTS as training
    args = SimpleNamespace(
        num_features=num_features,
        num_classes=num_classes,
        encoder="gat",       # From your training command
        decoder="gat",       # From your training command
        num_hidden=256,      # Default
        num_heads=4,         # Default
        num_out_heads=1,     # Default
        num_layers=2,        # Default
        attn_drop=0.1,       # Default
        in_drop=0.2,         # Default
        residual=False,      # Default
        norm=None,           # Default
        negative_slope=0.2,  # Default
        activation="prelu",  # Default
        mask_rate=0.75,      # Default
        replace_rate=0.0,    # Default
        alpha_l=2,           # Default
        loss_fn="sce",       # Default
        concat_hidden=False, # Default

        # --- ADD THESE MISSING DEFAULTS ---
        drop_edge_rate=0.0,  # Default
        pooling="mean",      # Default
        deg4feat=False       # Default
        # ------------------------------------
    )

    # 2. Build the model
    model = build_model(args)

    # 3. Load the saved weights
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    print("--- Model loaded successfully ---")
    return model

def get_subgraph_embedding(full_graph, model, start_node_idx, k_hop, device):
    """
    Samples a k-hop subgraph and computes its mean-pooled embedding.
    """
    # 1. Get the k-hop subgraph
    # subset: node indices from full_graph
    # new_edge_index: re-indexed edges for the subgraph
    # mapping: index in subgraph -> index in full_graph
    # edge_mask: which edges were kept
    start_node_tensor = torch.tensor([start_node_idx])
    subset, new_edge_index, mapping, edge_mask = k_hop_subgraph(
        start_node_tensor, k_hop, full_graph.edge_index, relabel_nodes=True
    )

    # 2. Get features for the subgraph nodes
    x = full_graph.x[subset].to(device)
    new_edge_index = new_edge_index.to(device)

    # 3. Create a batch vector for pooling
    # All nodes belong to the same graph (batch 0)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    # 4. Get node embeddings
    with torch.no_grad():
        # model.embed gets the node features from the encoder
        node_embeddings = model.embed(x, new_edge_index)

    # 5. Pool node embeddings to get a single graph embedding
    # We average all node embeddings to get one vector
    graph_embedding = global_mean_pool(node_embeddings, batch)

    return graph_embedding.cpu().numpy().flatten()


def main():
    parser = argparse.ArgumentParser(description="Visualize subgraph embeddings with t-SNE")
    parser.add_argument("--graph_path", type=str, default="graph_res7.pt", help="Path to the original .pt graph file")
    parser.add_argument("--model_path", type=str, default="checkpoint.pt", help="Path to the trained model checkpoint")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of subgraphs to sample")
    parser.add_argument("--k_hop", type=int, default=2, help="Neighborhood size (k-hop) for each subgraph")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Graph
    graph = load_graph_data(args.graph_path)

    # 2. Load Model
    model = load_trained_model(
        args.model_path,
        num_features=graph.num_node_features,
        num_classes=int(graph.y.max().item()) + 1
    ).to(device)

    # 3. Sample subgraphs and get embeddings
    embedding_list = []
    
    # Choose random start nodes for our subgraphs
    start_nodes = np.random.choice(graph.num_nodes, args.num_samples, replace=True)

    print(f"--- Generating {args.num_samples} subgraph embeddings... ---")
    for start_node in tqdm(start_nodes):
        emb = get_subgraph_embedding(graph, model, start_node, args.k_hop, device)
        embedding_list.append(emb)

    # 4. Run t-SNE
    print("--- Running t-SNE... ---")
    X = np.array(embedding_list)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)

    # 5. Plot results
    print("--- Plotting... ---")
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=20)
    plt.title(f"t-SNE of {args.num_samples} {args.k_hop}-Hop Subgraph Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()