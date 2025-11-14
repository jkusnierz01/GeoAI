import logging
import numpy as np
from tqdm import tqdm
import torch
import warnings

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


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(x, graph.edge_index)

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
    original_graph_file = "graph_res7.pt" 
    graph = torch.load(original_graph_file, map_location="cpu", weights_only=False)

    num_nodes = graph.num_nodes
    if not hasattr(graph, 'y') or graph.y is None:
        print("--- Adding dummy labels (y) ---")
        graph.y = torch.zeros(num_nodes, dtype=torch.long)
    
    if not hasattr(graph, 'train_mask'):
        print("--- Adding dummy masks (train/val/test) ---")
        graph.train_mask = torch.ones(num_nodes, dtype=torch.bool)
        graph.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        graph.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    if graph.edge_index.shape[0] == 4:
        warnings.warn(f"Graph's edge_index has shape {graph.edge_index.shape}. Slicing to [2, N] (taking first two rows).")
        graph.edge_index = graph.edge_index[:2, :] 
    elif graph.edge_index.shape[0] != 2:
        raise ValueError(f"Graph's edge_index has unsupported shape: {graph.edge_index.shape}. Expected [2, N].")

    num_features = graph.num_node_features
    num_classes = int(graph.y.max().item()) + 1  # This now works
    print(f"--- Graph prepared: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges. ---")
    print(f"--- Found {num_features} features and {num_classes} classes. ---")

    args.num_features = num_features
    args.num_classes = num_classes
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
            
        x = graph.x
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
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
