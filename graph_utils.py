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


def load_model_from_checkpoint(checkpoint_path, num_features, num_classes):
    """
    Build model with correct defaults and load checkpoint weights.
    """
    args = SimpleNamespace(
        num_features=num_features,
        num_classes=num_classes,

        # Encoder/decoder defaults
        encoder="gat",
        decoder="gat",

        # Architecture defaults
        num_hidden=256,
        num_heads=4,
        num_out_heads=1,
        num_layers=2,
        attn_drop=0.1,
        in_drop=0.2,
        residual=False,
        norm=None,
        negative_slope=0.2,
        activation="prelu",
        mask_rate=0.75,
        replace_rate=0.0,
        alpha_l=2,
        loss_fn="sce",
        concat_hidden=False,
        drop_edge_rate=0.0,
        pooling="mean",
        deg4feat=False,
    )

    model = build_model(args)

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print("--- Model successfully loaded ---")
    return model
