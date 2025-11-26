import numpy as np
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm

import os
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import k_hop_subgraph

from utils.graph_utils import (
    load_graphs_from_folder,
    prepare_graph,
)
from utils.model_utils import load_model_from_checkpoint, get_k_hop_subgraph_embedding
from utils.file_utils import get_prefix

def compute_isotropy_metrics(embeddings):
    """
    Computes the eigenvalue spectrum and the Participation Ratio (effective dimension)
    of the embedding space.
    """
    # 1. Center the embeddings
    mu = np.mean(embeddings, axis=0)
    X = embeddings - mu

    # 2. Compute Eigenvalues using SVD (numerically stable)
    #    Singular values s from SVD(X) are related to eigenvalues of Covariance by: lambda = s^2 / (N-1)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    eigenvalues = (s ** 2) / (X.shape[0] - 1)

    # 3. Calculate Participation Ratio (Effective Dimension)
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)
    
    if sum_eig_sq == 0:
        participation_ratio = 0.0
    else:
        participation_ratio = (sum_eig ** 2) / sum_eig_sq

    # 4. Simple Isotropy Measure (Min/Max ratio)
    isotropy_score = eigenvalues.min() / (eigenvalues.max() + 1e-9)

    return {
        "eigenvalues": eigenvalues,
        "participation_ratio": participation_ratio,
        "isotropy_score": isotropy_score
    }

def plot_eigenvalue_spectrum(metrics, title_suffix="", save_path="eigenvalue_spectrum.png"):
    """
    Plots the eigenvalue spectrum on a log scale.
    """
    eigenvalues = metrics["eigenvalues"]
    pr = metrics["participation_ratio"]
    iso = metrics["isotropy_score"]

    # Explained Variance Ratio
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / (total_variance + 1e-9)

    plt.figure(figsize=(10, 6))
    
    # Log-Log plot
    plt.plot(range(1, len(eigenvalues) + 1), explained_variance, marker='.', linestyle='-', linewidth=1.5)
    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('Principal Component Index (log)')
    plt.ylabel('Explained Variance Ratio (log)')
    plt.title(f'Anisotropy Analysis {title_suffix}\nEffective Dim (PR): {pr:.2f} / {len(eigenvalues)}')
    
    # Text box with metrics
    textstr = '\n'.join((
        f'Effective Dim: {pr:.2f}',
        f'Isotropy (Min/Max): {iso:.4f}',
        f'Top-1 Var: {explained_variance[0]:.2%}',
        f'Top-5 Var: {np.sum(explained_variance[:5]):.2%}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='bottom', bbox=props)

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved spectrum plot to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Calculate Anisotropy/Eigen-spectrum of Embeddings")
    parser.add_argument("--dataset", type=str, required=True, help="Path to folder with .pt files")
    parser.add_argument("--model_path", type=str, default="checkpoint.pt", help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, default="configs/defaults.yaml", help="Path to model config")
    parser.add_argument("--samples_per_graph", type=int, default=50, help="Number of subgraphs to sample per graph file")
    parser.add_argument("--k_hop", type=int, default=2, help="K-hop size for subgraph sampling")
    parser.add_argument("--output_plot", type=str, default="isotropy_spectrum.png", help="Filename for the output plot")
    
    args = parser.parse_args()

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    print(f"Loading graphs from {args.dataset}...")
    graph_files = load_graphs_from_folder(args.dataset)
    # Prepare graphs (loading .pt files into PyTorch Geometric objects)
    graphs = [prepare_graph(f) for f in graph_files]
    
    if not graphs:
        print("No graphs found. Exiting.")
        return

    # 3. Load Model
    # Determine input/output dims from data
    num_features = graphs[0].num_node_features
    # Determine num_classes based on max y label found in dataset
    num_classes = max(g.y.max().item() for g in graphs) + 1

    model = load_model_from_checkpoint(
        args.model_path, 
        num_features, 
        num_classes, 
        config_path=args.config_path
    ).to(device)

    # 4. Generate Embeddings
    embeddings = []
    
    print(f"--- Sampling {args.samples_per_graph} subgraphs per file ---")
    total_files = len(graphs)
    
    for file_idx, graph in enumerate(graphs):
        # Optional: Print progress per file
        # print(f"Processing {file_idx+1}/{total_files}...")
        
        num_nodes = graph.num_nodes
        
        # Randomly sample start nodes for subgraphs
        start_nodes = np.random.choice(
            num_nodes, args.samples_per_graph, replace=True
        )

        for s in start_nodes:
            emb = get_k_hop_subgraph_embedding(graph, model, s, args.k_hop, device)
            embeddings.append(emb)

    embeddings = np.array(embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # 5. Calculate Isotropy & Plot
    print("--- Calculating Eigenvalue Spectrum ---")
    metrics = compute_isotropy_metrics(embeddings)
    
    print(f"Effective Dimension (PR): {metrics['participation_ratio']:.4f}")
    print(f"Isotropy (Min/Max):       {metrics['isotropy_score']:.6f}")

    plot_eigenvalue_spectrum(
        metrics, 
        title_suffix=f"(Dataset: {os.path.basename(args.dataset)})", 
        save_path=args.output_plot
    )
    print("Done.")

if __name__ == "__main__":
    main()