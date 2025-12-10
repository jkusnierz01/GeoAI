import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch_geometric.nn import global_mean_pool

from utils.graph_utils import load_graphs_from_folder, prepare_graph
from utils.model_utils import load_model_from_checkpoint, get_k_hop_subgraph_embedding


def covering_khop_for_resolution(k_hop_low_res, delta_levels):
    """
    Map k-hop from lower-resolution (larger area) to higher-resolution (smaller area).
    delta_levels > 0 means going up in resolution (smaller area),
    delta_levels < 0 means going down in resolution (larger area).
    """
    if delta_levels <= 0:
        return max(1, int(k_hop_low_res))
    factor = 2 ** delta_levels  # sqrt(area scaling)
    return max(1, int(np.ceil(k_hop_low_res * factor)))


def cosine_sim(a, b):
    """Numerically stable cosine similarity between 1D vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.dot(a, b)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(num / den)


def process_dataset(dataset_path, model, device, samples_per_graph, base_k_hop):
    """
    Sample nodes per graph and compute multi-resolution embeddings.
    Starts from lowest resolution (largest area) and goes up.
    """
    results = []

    # Define k-hops for each resolution
    k7 = base_k_hop              # res7: lowest resolution
    k8 = covering_khop_for_resolution(k7, delta_levels=1)
    k9 = covering_khop_for_resolution(k7, delta_levels=2)

    print(f"Using k-hops: res7={k7}, res8={k8}, res9={k9}")

    graph_files = load_graphs_from_folder(dataset_path)
    graphs = [prepare_graph(f) for f in graph_files]

    for g_idx, graph in enumerate(tqdm(graphs, desc="graphs")):
        if graph.num_nodes == 0:
            continue

        centers = np.random.choice(graph.num_nodes, samples_per_graph, replace=True)

        for center in centers:
            try:
                # Compute embeddings
                emb_7 = get_k_hop_subgraph_embedding(graph, model, center, k7, device)
                emb_8 = get_k_hop_subgraph_embedding(graph, model, center, k8, device)
                emb_9 = get_k_hop_subgraph_embedding(graph, model, center, k9, device)

                # Cross-resolution similarities
                sim_7_8 = cosine_sim(emb_7, emb_8)
                sim_7_9 = cosine_sim(emb_7, emb_9)
                sim_8_9 = cosine_sim(emb_8, emb_9)

                results.append({
                    "graph_idx": int(g_idx),
                    "center": int(center),
                    "mode": "multi_res",
                    "k_res7": int(k7),
                    "k_res8": int(k8),
                    "k_res9": int(k9),
                    "sim_res7_res8": sim_7_8,
                    "sim_res7_res9": sim_7_9,
                    "sim_res8_res9": sim_8_9,
                })
            except Exception as e:
                print(f"Warning: graph {g_idx}, center {center} failed: {e}")

    return results


def process_random_pairs(graphs, model, device, random_pairs_per_graph, base_k_hop):
    """Random independent node pairs to measure embedding similarity at each resolution."""
    results = []

    k7 = base_k_hop
    k8 = covering_khop_for_resolution(k7, delta_levels=1)
    k9 = covering_khop_for_resolution(k7, delta_levels=2)

    for g_idx, graph in enumerate(tqdm(graphs, desc="random_pairs")):
        if graph.num_nodes < 2:
            continue

        nodes_a = np.random.choice(graph.num_nodes, random_pairs_per_graph, replace=True)
        nodes_b = np.random.choice(graph.num_nodes, random_pairs_per_graph, replace=True)

        for idx_a, idx_b in zip(nodes_a, nodes_b):
            try:
                emb_a7 = get_k_hop_subgraph_embedding(graph, model, idx_a, k7, device)
                emb_a8 = get_k_hop_subgraph_embedding(graph, model, idx_a, k8, device)
                emb_a9 = get_k_hop_subgraph_embedding(graph, model, idx_a, k9, device)

                emb_b7 = get_k_hop_subgraph_embedding(graph, model, idx_b, k7, device)
                emb_b8 = get_k_hop_subgraph_embedding(graph, model, idx_b, k8, device)
                emb_b9 = get_k_hop_subgraph_embedding(graph, model, idx_b, k9, device)

                results.append({
                    "graph_idx": int(g_idx),
                    "node_a": int(idx_a),
                    "node_b": int(idx_b),
                    "mode": "random_pairs",
                    "sim_res7": cosine_sim(emb_a7, emb_b7),
                    "sim_res8": cosine_sim(emb_a8, emb_b8),
                    "sim_res9": cosine_sim(emb_a9, emb_b9),
                    "sim_7_8": cosine_sim(emb_a7, emb_b8),
                    "sim_7_9": cosine_sim(emb_a7, emb_b9),
                    "sim_8_9": cosine_sim(emb_a8, emb_b9),
                })
            except Exception as e:
                print(f"Warning random pairs: graph {g_idx}, pair({idx_a},{idx_b}) failed: {e}")

    return results


def summarize_stats(results, output_file="similarity_stats.txt"):
    """Compute aggregated statistics for all modes."""
    stats = defaultdict(list)
    for r in results:
        if r.get("mode") == "multi_res":
            stats["7-8"].append(r.get("sim_res7_res8"))
            stats["7-9"].append(r.get("sim_res7_res9"))
            stats["8-9"].append(r.get("sim_res8_res9"))
        elif r.get("mode") == "random_pairs":
            stats["random_7"].append(r.get("sim_res7"))
            stats["random_8"].append(r.get("sim_res8"))
            stats["random_9"].append(r.get("sim_res9"))
            stats["random_7-8"].append(r.get("sim_7_8"))
            stats["random_7-9"].append(r.get("sim_7_9"))
            stats["random_8-9"].append(r.get("sim_8_9"))

    def agg(arr):
        arr = np.array([x for x in arr if x is not None])
        if arr.size == 0:
            return None
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max())
        }

    print("\n=== Aggregated similarity stats ===")
    with open(output_file, "w") as f:
        for k in sorted(stats.keys()):
            a = agg(stats[k])
            if a is None:
                print(f"{k}: no samples")
                f.write(f"{k}: no samples\n")
            else:
                print(f"{k}: n={a['count']}, mean={a['mean']:.4f}, std={a['std']:.4f}, min={a['min']:.4f}, max={a['max']:.4f}")
                f.write(f"{k}: n={a['count']}, mean={a['mean']:.4f}, std={a['std']:.4f}, min={a['min']:.4f}, max={a['max']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-resolution cosine similarity on graph dataset")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="checkpoint.pt")
    parser.add_argument("--config_path", type=str, default="configs/defaults.yaml")
    parser.add_argument("--samples_per_graph", type=int, default=50)
    parser.add_argument("--base_k_hop", type=int, default=1, help="k-hop for lowest resolution (res7)")
    parser.add_argument("--random_pairs_per_graph", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    graph_files = load_graphs_from_folder(args.dataset)
    graphs = [prepare_graph(f) for f in graph_files]
    if not graphs:
        print("No graphs found. Exiting.")
        return

    num_features = graphs[0].num_node_features
    num_classes = max(g.y.max().item() for g in graphs) + 1

    model = load_model_from_checkpoint(
        args.model_path, num_features, num_classes,
        config_path=args.config_path, device=device
    )

    results = process_dataset(args.dataset, model, device, args.samples_per_graph, args.base_k_hop)
    results.extend(process_random_pairs(graphs, model, device, args.random_pairs_per_graph, args.base_k_hop))

    summarize_stats(results)
    print("Done.")


if __name__ == "__main__":
    main()
