import torch
import pytorch_lightning as pl
import numpy as np
from types import SimpleNamespace
from src.graphmae.models import build_model
from src.graphmae.utils import create_optimizer

class GraphMAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        args = SimpleNamespace(**self.hparams)
        
        self.model = build_model(args)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.model(batch.x, batch.edge_index)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(
            self.hparams.optimizer,
            self.model,
            self.hparams.lr,
            self.hparams.weight_decay
        )

        if self.hparams.scheduler:
            max_epoch = self.hparams.max_epoch
            scheduler_lambda = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)
            return [optimizer], [scheduler]
        
        return optimizer
