import argparse
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Added for custom colormap
from sklearn.manifold import TSNE
import rootutils

ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.utils.graph_utils import (
    load_graphs_from_folder,
    prepare_graph,
)

from src.utils.model_utils import get_k_hop_subgraph_embedding
from src.utils.file_utils import get_prefix

from src.models.graphmae_module import GraphMAE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to folder with .pt files")
    parser.add_argument("--model_path", type=str, default="checkpoint.pt")
    parser.add_argument("--samples_per_graph", type=int, default=100)
    parser.add_argument("--k_hop", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_files = load_graphs_from_folder(args.dataset)
    graphs = [prepare_graph(f) for f in graph_files]

    prefixes = [get_prefix(f) for f in graph_files]
    unique_prefixes = sorted(set(prefixes))
    prefix_to_color_id = {p: i for i, p in enumerate(unique_prefixes)}

    print(f"Loading model from {args.model_path}...")
    model = GraphMAE.load_from_checkpoint(args.model_path, weights_only=False)
    model.to(device)
    model.eval()

    embeddings = []
    color_ids = []
    file_groups = []

    print("--- Sampling and embedding subgraphs ---")
    for file_idx, graph in enumerate(graphs):
        prefix = prefixes[file_idx]
        color_id = prefix_to_color_id[prefix]

        num_nodes = graph.num_nodes
        # Guard against sampling more nodes than exist
        actual_samples = min(num_nodes, args.samples_per_graph)
        start_nodes = np.random.choice(num_nodes, actual_samples, replace=False)

        for s in tqdm(start_nodes, desc=f"{prefix}"):
            try:
                emb = get_k_hop_subgraph_embedding(graph, model, s, args.k_hop, device)
            except IndexError as e:
                print(f"[Error] IndexError for node {s} in graph {prefix}: {e}. Skipping.")
                continue
            embeddings.append(emb)
            color_ids.append(color_id)
            file_groups.append(prefix)

    embeddings = np.array(embeddings)
    color_ids = np.array(color_ids)

    print("--- Running t-SNE ---")
    X_2d = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42
    ).fit_transform(embeddings)

    print("--- Plotting ---")
    
    # --- CUSTOM COLOR LOGIC START ---
    # 1. Concatenate two 20-color palettes to get 40 distinct colors
    colors1 = plt.cm.tab20.colors
    colors2 = plt.cm.tab20b.colors
    combined_colors = colors1 + colors2
    
    # 2. Create the custom colormap
    custom_cmap = mcolors.ListedColormap(combined_colors)
    
    # 3. Determine range for normalization so index 0 maps to color 0, index 5 to color 5, etc.
    # We set vmax to len(combined_colors) - 1 so the mapping is exact.
    max_color_idx = len(combined_colors) - 1
    # --- CUSTOM COLOR LOGIC END ---

    plt.figure(figsize=(12, 10)) # Increased width for the larger legend
    
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=color_ids,
        cmap=custom_cmap,
        vmin=0,                 # Fix the lower bound of color mapping
        vmax=max_color_idx,     # Fix the upper bound of color mapping
        alpha=0.7,
        s=20
    )

    plt.title(f"t-SNE of Subgraph Embeddings")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")

    handles = []
    labels = []

    def pretty_city_name(prefix):
        # Remove trailing _hexagons(_resX) if present
        name = prefix
        if name.endswith('_hexagons'):
            name = name[:-9]
        elif '_hexagons_res' in name:
            name = name[:name.index('_hexagons_res')]
        # Replace underscores with spaces and capitalize each word
        return ' '.join([w.capitalize() for w in name.split('_')])

    # Create legend handles manually to ensure they match the scatter colors
    for prefix, cid in prefix_to_color_id.items():
        # Safely wrap around if you somehow exceed 40 (though you have 35)
        c_val = combined_colors[cid % len(combined_colors)]
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="", color=c_val, markersize=8)
        )
        labels.append(pretty_city_name(prefix))

    # ncol=2 splits the long list of 35 cities into two columns
    plt.legend(
        handles, 
        labels, 
        title="Graph Groups", 
        loc="center left", 
        bbox_to_anchor=(1.02, 0.5),
        ncol=2,
        fontsize='small'
    )
    
    output_filename = "tsne_embeddings_35_cities.png"
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_filename}")

if __name__ == "__main__":
    main()