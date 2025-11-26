import os
import torch
import warnings
from types import SimpleNamespace
from graphmae.models import build_model

def load_graphs_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Dataset folder does not exist: {folder_path}")

    files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".pt")
    ])

    if len(files) == 0:
        raise ValueError(f"No .pt files found in dataset folder: {folder_path}")

    print(f"Found {len(files)} graph files in folder: {folder_path}")
    return files


def prepare_graph(path):
    graph = torch.load(path, map_location="cpu", weights_only=False)

    if not hasattr(graph, "y") or graph.y is None:
        graph.y = torch.zeros(graph.num_nodes, dtype=torch.long)

    if not hasattr(graph, "train_mask"):
        graph.train_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
        graph.val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        graph.test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

    if graph.edge_index.shape[0] == 4:
        warnings.warn(f"Graph edge_index 4 rows → slicing to first 2")
        graph.edge_index = graph.edge_index[:2, :]
    elif graph.edge_index.shape[0] != 2:
        raise ValueError(f"Unsupported edge_index shape: {graph.edge_index.shape}")

    return graph
