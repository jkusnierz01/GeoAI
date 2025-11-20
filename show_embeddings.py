import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import k_hop_subgraph

from graph_utils import (
    load_graphs_from_folder,
    prepare_graph,
    load_model_from_checkpoint,
)


def get_prefix(filename):
    """
    For files like 'abc_res7.geojson' → return 'abc'.
    """
    base = os.path.basename(filename)
    if "_res" in base:
        return base.split("_res")[0]
    return base.split(".")[0]


def get_subgraph_embedding(full_graph, model, start_node_idx, k_hop, device):
    start_node_tensor = torch.tensor([start_node_idx])
    subset, edge_index, _, _ = k_hop_subgraph(
        start_node_tensor, k_hop, full_graph.edge_index, relabel_nodes=True
    )

    x = full_graph.x[subset].to(device)
    edge_index = edge_index.to(device)

    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        node_embeds = model.embed(x, edge_index)

    graph_embed = global_mean_pool(node_embeds, batch)
    return graph_embed.cpu().numpy().flatten()


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

    num_features = graphs[0].num_node_features
    num_classes = max(g.y.max().item() for g in graphs) + 1

    model = load_model_from_checkpoint(
        args.model_path, num_features, num_classes
    ).to(device)

    embeddings = []
    color_ids = []
    file_groups = []

    print("--- Sampling and embedding subgraphs ---")
    for file_idx, graph in enumerate(graphs):
        prefix = prefixes[file_idx]
        color_id = prefix_to_color_id[prefix]

        num_nodes = graph.num_nodes

        start_nodes = np.random.choice(
            num_nodes, args.samples_per_graph, replace=True
        )

        for s in tqdm(start_nodes, desc=f"{prefix}"):
            emb = get_subgraph_embedding(graph, model, s, args.k_hop, device)
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
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=color_ids,
        cmap="tab20",
        alpha=0.7,
        s=20
    )

    plt.title("t-SNE of Subgraph Embeddings (colored by graph group)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")

    handles = []
    labels = []

    for prefix, cid in prefix_to_color_id.items():
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="", color=plt.cm.tab20(cid), markersize=8)
        )
        labels.append(prefix)

    plt.legend(handles, labels, title="Graph Groups", loc="best")

    plt.show()


if __name__ == "__main__":
    main()
