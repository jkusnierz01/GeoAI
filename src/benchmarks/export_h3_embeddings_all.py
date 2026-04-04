import argparse
import pickle
from pathlib import Path
import sys
from typing import Iterable, List, Optional

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.graphmae_module import GraphMAE
from src.utils.graph_utils import prepare_graph
from src.utils.model_utils import load_model_from_checkpoint


def _iter_graph_files(directories: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for directory in directories:
        if not directory.exists():
            print(f"[warn] Directory not found, skipping: {directory}")
            continue
        files.extend(sorted(directory.glob("*.pt")))
    return files


def _extract_h3_ids(graph) -> Optional[List[str]]:
    candidates = ["h3_ids", "node_names", "h3_index", "h3_indexes", "h3_indices"]
    raw = None
    for attr in candidates:
        if hasattr(graph, attr):
            raw = getattr(graph, attr)
            break

    if raw is None:
        return None

    if isinstance(raw, torch.Tensor):
        raw = raw.cpu().tolist()

    if not isinstance(raw, list):
        try:
            raw = list(raw)
        except TypeError:
            return None

    # Some graphs may store nested singleton lists; flatten one level if needed.
    if raw and isinstance(raw[0], list):
        if len(raw) == 1 and isinstance(raw[0], list):
            raw = raw[0]
        elif all(isinstance(item, list) and len(item) == 1 for item in raw):
            raw = [item[0] for item in raw]

    return [str(v) for v in raw]


def _embed_graph_nodes(graph, model, device: torch.device):
    x = graph.x.to(device)
    edge_index = graph.edge_index[:2].to(device)

    with torch.no_grad():
        if hasattr(model, "embed"):
            node_embeds = model.embed(x, edge_index)
        else:
            node_embeds = model.model.embed(x, edge_index)

    return node_embeds.cpu().numpy()


def _load_model(model_path: Path, config_path: Path, sample_graph, device: torch.device):
    # Try Lightning checkpoint first (GraphMAE.load_from_checkpoint), then fallback.
    try:
        model = GraphMAE.load_from_checkpoint(str(model_path), weights_only=False)
        model.to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"[warn] GraphMAE checkpoint load failed, using fallback loader: {exc}")

    try:
        num_classes = int(sample_graph.y.max().item()) + 1 if hasattr(sample_graph, "y") else 0
    except Exception:
        num_classes = 0

    model = load_model_from_checkpoint(
        str(model_path),
        sample_graph.num_node_features,
        num_classes,
        config_path=str(config_path),
        device=device,
    )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export H3 -> Graph embedding dictionaries for all graphs in selected directories"
    )
    parser.add_argument("--model_path", type=str, default=str(ROOT / "checkpoints/plain.ckpt"))
    parser.add_argument("--config_path", type=str, default=str(ROOT / "configs/defaults.yaml"))
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[
            str(ROOT / "dataset_aligned"),
            str(ROOT / "src/benchmarks/graphs"),
            str(ROOT / "src/benchmarks/graph_for_satelite"),
        ],
        help="Directories containing .pt graph files",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT / "src/benchmarks/embeddings/h3_graph_embeddings"),
        help="Output root for .pkl files",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)
    config_path = Path(args.config_path)
    directories = [Path(p) for p in args.dirs]
    output_root = Path(args.output_root)

    graph_files = _iter_graph_files(directories)
    print(f"Found {len(graph_files)} graph files.")
    if not graph_files:
        return

    sample_graph = prepare_graph(str(graph_files[0]))
    model = _load_model(model_path, config_path, sample_graph, device)

    saved = 0
    skipped = 0

    for graph_file in tqdm(graph_files, desc="Exporting H3 embeddings"):
        try:
            graph = prepare_graph(str(graph_file))
            h3_ids = _extract_h3_ids(graph)
            if not h3_ids:
                print(f"[warn] Missing H3 IDs in {graph_file}, skipping.")
                skipped += 1
                continue

            node_embeds = _embed_graph_nodes(graph, model, device)

            n = min(len(h3_ids), node_embeds.shape[0])
            if n == 0:
                print(f"[warn] No rows to save for {graph_file}, skipping.")
                skipped += 1
                continue
            if len(h3_ids) != node_embeds.shape[0]:
                print(
                    f"[warn] Length mismatch in {graph_file}: "
                    f"h3_ids={len(h3_ids)} vs embeds={node_embeds.shape[0]}; using first {n}."
                )

            data = {h3_ids[i]: node_embeds[i] for i in range(n)}

            rel = graph_file.relative_to(ROOT)
            out_file = output_root / rel.parent / f"{graph_file.stem}_h3_embeddings.pkl"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, "wb") as f:
                pickle.dump(data, f)

            saved += 1
        except Exception as exc:
            print(f"[warn] Failed {graph_file}: {exc}")
            skipped += 1

    print(f"Done. Saved: {saved}, Skipped: {skipped}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
