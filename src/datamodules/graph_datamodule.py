import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from src.utils.graph_utils import load_graphs_from_folder, prepare_graph


class GraphDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        graph_files = load_graphs_from_folder(dataset_path)
        self.graphs = [prepare_graph(f) for f in graph_files]

    def __getitem__(self, index):
        return self.graphs[index]

    def __len__(self):
        return len(self.graphs)


class GraphDataModule(L.LightningDataModule):
    def __init__(self, dataset_path: str, num_workers: int, batch_size: int) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        ...

    def train_dataloader(self):
        return DataLoader(GraphDataset(self.dataset_path), batch_size=self.batch_size, num_workers=self.num_workers)
