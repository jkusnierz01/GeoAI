import logging
import numpy as np
from tqdm import tqdm
import os
import torch
import warnings
from torch_geometric.loader import DataLoader

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def load_graphs_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Dataset folder does not exist: {folder_path}")

    # Collect .pt files
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

    num_nodes = graph.num_nodes

    # Ensure y
    if not hasattr(graph, 'y') or graph.y is None:
        graph.y = torch.zeros(num_nodes, dtype=torch.long)

    # Ensure train/val/test masks
    if not hasattr(graph, 'train_mask'):
        graph.train_mask = torch.ones(num_nodes, dtype=torch.bool)
        graph.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        graph.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Fix edge_index shape
    if graph.edge_index.shape[0] == 4:
        warnings.warn(f"Graph's edge_index has shape {graph.edge_index.shape}. Slicing to [2, N] (taking first two rows).")
        graph.edge_index = graph.edge_index[:2, :]
    elif graph.edge_index.shape[0] != 2:
        raise ValueError(
            f"Graph's edge_index has unsupported shape: {graph.edge_index.shape}. "
            "Expected [2, N]."
        )

    return graph


def pretrain(model, graphs, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        for batch in loader:
            model.train()
            loss, loss_dict = model(batch.x.to(device), batch.edge_index.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)

    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    # Load custom dataset - START
    dataset_path = args.dataset
    graph_files = load_graphs_from_folder(dataset_path)
    graphs = [prepare_graph(f) for f in graph_files]
    args.num_classes = num_classes = max(graph.y.max().item() for graph in graphs) + 1
    args.num_features = graphs[0].num_node_features
    # Load custom dataset - END

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        if not load_model:
            model = pretrain(model, graphs, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()

        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
