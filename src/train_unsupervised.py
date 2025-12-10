import logging
import numpy as np
from tqdm import tqdm
import hydra
import torch
from torch_geometric.loader import DataLoader

from utils.graph_utils import (
    load_graphs_from_folder,
    prepare_graph,
    build_model,
)

from omegaconf import OmegaConf, DictConfig
import rootutils
import wandb

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.models import build_model


ROOT = rootutils.setup_root(search_from=".", indicator=".project_root", pythonpath=True)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def pretrain(
    model,
    graphs,
    optimizer,
    max_epoch,
    device,
    scheduler,
    num_classes,
    lr_f,
    weight_decay_f,
    max_epoch_f,
    linear_prob,
    logger=None,
):
    logging.info("start training..")
    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    epoch_iter = tqdm(range(max_epoch))

    
    for epoch in epoch_iter:
        epoch_losses = [] 
        for batch in loader:
            model.train()
            loss, loss_dict = model(batch.x.to(device), batch.edge_index.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item()) 

        if scheduler is not None:
            scheduler.step()
            
        avg_loss = np.mean(epoch_losses)
        
        epoch_iter.set_description(
            f"# Epoch {epoch}: train_loss: {avg_loss:.4f}"
        )
        
        loss_dict["loss"] = avg_loss
        loss_dict["lr"] = get_current_lr(optimizer)
        
        if logger is not None:
            logger.note(loss_dict, step=epoch)
        if wandb.run is not None:
            wandb.log(loss_dict, step=epoch)

    return model


@hydra.main(version_base="1.1", config_path="configs", config_name="defaults.yaml")
def main(cfg: DictConfig) -> None:
    device = cfg.device if cfg.device >= 0 else "cpu"
    device_name = (
        torch.cuda.get_device_name(device)
        if torch.cuda.is_available() and device != "cpu"
        else "CPU"
    )
    print(f"Using device: {device_name}")

    seeds = cfg.seeds
    dataset_name = cfg.dataset
    max_epoch = cfg.max_epoch
    max_epoch_f = cfg.max_epoch_f
    num_hidden = cfg.num_hidden
    num_layers = cfg.num_layers
    encoder_type = cfg.encoder
    decoder_type = cfg.decoder
    replace_rate = cfg.replace_rate

    optim_type = cfg.optimizer
    loss_fn = cfg.loss_fn

    lr = cfg.lr
    weight_decay = cfg.weight_decay
    lr_f = cfg.lr_f
    weight_decay_f = cfg.weight_decay_f
    linear_prob = cfg.linear_prob
    load_model = cfg.load_model
    save_model = cfg.save_model
    logs = cfg.logging
    use_scheduler = cfg.scheduler

    dataset_path = ROOT / cfg.dataset
    graph_files = load_graphs_from_folder(dataset_path)
    graphs = [prepare_graph(f) for f in graph_files]
    num_classes = max(graph.y.max().item() for graph in graphs) + 1
    num_features = graphs[0].num_node_features

    OmegaConf.set_struct(cfg, False)
    cfg.num_features = num_features

    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if cfg.wandb:
            wandb.init(
                entity="pgm-team",
                project="GeoAI",
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
                name=f"GraphMAE_{cfg.dataset}_{cfg.encoder}_lr:{lr}",
                reinit=True
            )

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}"
            )
        else:
            logger = None

        model = build_model(cfg)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=scheduler
            )
        else:
            scheduler = None

        if not load_model:
            model = pretrain(
                model,
                graphs,
                optimizer,
                max_epoch,
                device,
                scheduler,
                num_classes,
                lr_f,
                weight_decay_f,
                max_epoch_f,
                linear_prob,
                logger,
            )
            model = model.cpu()

        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        if logger is not None:
            logger.finish()

        if cfg.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
