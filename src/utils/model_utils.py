import os
from omegaconf import DictConfig
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import k_hop_subgraph
from src.utils.graph_utils import (
    build_model
)
import hydra

def load_model_from_checkpoint(checkpoint_path, num_features, num_classes, config_path="configs/defaults.yaml", device="cpu"):
    """
    Build model with parameters from config and load checkpoint weights.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    cfg = OmegaConf.load(config_path)

    args = SimpleNamespace(
        num_features=num_features,
        num_classes=num_classes,
        encoder=cfg.encoder,
        decoder=cfg.decoder,
        num_hidden=cfg.num_hidden,
        num_heads=cfg.num_heads,
        num_out_heads=cfg.num_out_heads,
        num_layers=cfg.num_layers,
        attn_drop=cfg.attn_drop,
        in_drop=cfg.in_drop,
        residual=cfg.residual,
        norm=cfg.norm,
        negative_slope=cfg.negative_slope,
        activation=cfg.activation,
        mask_rate=cfg.mask_rate,
        replace_rate=cfg.replace_rate,
        alpha_l=cfg.alpha_l,
        loss_fn=cfg.loss_fn,
        concat_hidden=cfg.concat_hidden,
        drop_edge_rate=cfg.drop_edge_rate,
        pooling="mean",
        deg4feat=False,
    )

    model = build_model(args).to(device)
    import numpy as np
    torch.serialization.add_safe_globals([
        np.dtype,
        __import__('numpy')._core.multiarray.scalar,
        __import__('numpy').dtypes.Float64DType
    ])
    state = torch.load(checkpoint_path, map_location=device)
    
    # Handle checkpoints with different formats
    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
        sd = state.get('model_state_dict', state.get('state_dict', state))
        # Remove 'model.' prefix if present in state_dict keys
        sd = {k.replace('model.', '', 1) if k.startswith('model.') else k: v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        model.load_state_dict(state)

    model.eval()
    print(f"--- Model loaded from {checkpoint_path} ---")
    return model

def get_k_hop_subgraph_embedding(full_graph, model, start_node_idx, k_hop, device):
    """
    Extract a k-hop subgraph around a node and return pooled embedding (numpy vector).
    Uses relabel_nodes=True to create a compact subgraph for model input.
    """
    start_node_tensor = torch.tensor([int(start_node_idx)], dtype=torch.long)
    subset, edge_index, _, _ = k_hop_subgraph(
        start_node_tensor, k_hop, full_graph.edge_index, relabel_nodes=True
    )

    x = full_graph.x[subset].to(device)
    edge_index = edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    # Only print debug info for the first node processed per graph
    if not hasattr(full_graph, '_debug_embedding_printed'):
        with torch.no_grad():
            node_embeds = model.model.embed(x, edge_index)
        graph_embed = global_mean_pool(node_embeds, batch)
        print(f"[DEBUG] Node embeddings (before pooling) for node {start_node_idx}: mean={node_embeds.mean().item():.4f}, std={node_embeds.std().item():.4f}")
        print(f"[DEBUG] Graph embedding (after pooling) for node {start_node_idx}: mean={graph_embed.mean().item():.4f}, std={graph_embed.std().item():.4f}")
        full_graph._debug_embedding_printed = True
        return graph_embed.cpu().numpy().flatten()
    else:
        with torch.no_grad():
            node_embeds = model.model.embed(x, edge_index)
        graph_embed = global_mean_pool(node_embeds, batch)
        return graph_embed.cpu().numpy().flatten()


def instantiate_callbacks(callbacks_cfg: DictConfig):
    callbacks = []

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks