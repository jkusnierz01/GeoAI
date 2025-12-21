import lightning as L
from torch.utils.data import Dataset
from src.utils.graph_utils import load_graphs_from_folder, prepare_graph
from torch_geometric.loader import DataLoader

class GraphDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        graph_files = load_graphs_from_folder(dataset_path)
        self.graphs = []
        expected_dim = None
        for f in graph_files:
            g = prepare_graph(f)
            if hasattr(g, 'x') and g.x is not None:
                if expected_dim is None:
                    expected_dim = g.x.shape[1]
                    print(f"[GraphDataset] Expected feature dim: {expected_dim}")
                if g.x.shape[1] != expected_dim:
                    print(f"[GraphDataset] Skipping {f}: feature dim {g.x.shape[1]} != expected {expected_dim}")
                    continue
            else:
                print(f"[GraphDataset] Skipping {f}: missing 'x' attribute or None")
                continue
            self.graphs.append(g)
        print(f"[GraphDataset] Loaded {len(self.graphs)} graphs with feature dim {expected_dim}")

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
        return DataLoader(GraphDataset(self.dataset_path), batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)