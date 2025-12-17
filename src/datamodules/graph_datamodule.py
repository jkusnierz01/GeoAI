import lightning as L
from torch.utils.data import Dataset
from src.utils.graph_utils import load_graphs_from_folder, prepare_graph
from torch_geometric.loader import DataLoader


from torch_geometric.data import Data
import torch

class GraphDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        graph_files = load_graphs_from_folder(dataset_path)
        graphs = [prepare_graph(f) for f in graph_files]
        # Concatenate all graphs into a single large graph
        # Assumes all graphs have the same feature dimension
        x_list = []
        edge_index_list = []
        y_list = []
        node_offset = 0
        for g in graphs:
            x_list.append(g.x)
            y_list.append(g.y)
            edge_index_list.append(g.edge_index + node_offset)
            node_offset += g.x.size(0)
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        self.graph = Data(x=x, edge_index=edge_index, y=y)

    def __getitem__(self, index):
        # Only one big graph
        return self.graph

    def __len__(self):
        return 1


class GraphDataModule(L.LightningDataModule):
    def __init__(self, dataset_path: str, num_workers: int, batch_size: int) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        ...

    def train_dataloader(self):
        return DataLoader(GraphDataset(self.dataset_path), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
