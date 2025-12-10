
import hydra
from omegaconf import DictConfig
import rootutils
from hydra.utils import instantiate
import wandb



ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.utils.graph_utils import load_graphs_from_folder, prepare_graph
from src.graphmae.utils import set_random_seed
from src.datamodules.graph_datamodule import GraphDataset
from src.utils.model_utils import instantiate_callbacks


@hydra.main(version_base="1.1", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:

    logger = instantiate(cfg.logger)
    datamodule = instantiate(cfg.data)
    
    temp_dataset = GraphDataset(cfg.data.dataset_path)
    num_features = temp_dataset[0].num_node_features
    
    model = instantiate(cfg.model, num_features=num_features)
    
    callbacks = instantiate_callbacks(cfg.callbacks)
    
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger = logger)
    trainer.fit(model, datamodule)
    
    wandb.finish()

if __name__ == "__main__":
    main()
