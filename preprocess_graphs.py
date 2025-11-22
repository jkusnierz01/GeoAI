import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Preprocess graph datasets to align features.")
    parser.add_argument("--input_dir", "-i", type=str, default="dataset", help="Input directory containing .pt files")
    parser.add_argument("--output_dir", "-o", type=str, default="dataset_aligned", help="Output directory for processed files")
    parser.add_argument("--debug", action="store_true", help="Print debug info (e.g. rejected columns)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in input_path.glob("*.pt")])
    if not files:
        print(f"No .pt files found in {input_path}")
        return

    print(f"Loading {len(files)} graphs from {input_path}...")
    graphs = []
    filenames = []
    for f in tqdm(files):
        try:
            data = torch.load(f, map_location="cpu", weights_only=False) 
            graphs.append(data)
            filenames.append(f.name)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not graphs:
        return

    has_col_names = all(hasattr(g, 'feature_columns') for g in graphs)
    
    if has_col_names:
        print("Feature columns found in all graphs. Aligning by name...")
        
        # Intersection of all feature sets
        common_columns = set(graphs[0].feature_columns)
        all_columns = set(graphs[0].feature_columns)

        for g in graphs[1:]:
            common_columns.intersection_update(g.feature_columns)
            all_columns.update(g.feature_columns)
        
        common_columns = sorted(list(common_columns))

        if args.debug:
            rejected = all_columns - set(common_columns)
            if rejected:
                print(f"\n[DEBUG] Rejected {len(rejected)} columns (not present in all graphs):")
                for col in sorted(list(rejected)):
                    print(f" - {col}")
            else:
                print("\n[DEBUG] No columns were rejected.")
            print("-" * 40)

        print(f"Found {len(common_columns)} common features.")
        
        if len(common_columns) == 0:
            print("Error: No common features found!")
            return


        for i, g in enumerate(graphs):
            col_to_idx = {name: idx for idx, name in enumerate(g.feature_columns)}
            indices_to_keep = [col_to_idx[col] for col in common_columns]
            
            # Select columns
            g.x = g.x[:, indices_to_keep]
            
            # Update metadata
            g.num_node_features = len(common_columns)
            g.feature_columns = common_columns
            
    else:
        print("Feature columns NOT found in all graphs (or inconsistent). Falling back to truncation to minimum size.")
        min_features = min(g.num_node_features for g in graphs)
        print(f"Minimum feature count: {min_features}")
        
        for i, g in enumerate(graphs):
            if g.num_node_features > min_features:
                g.x = g.x[:, :min_features]
                g.num_node_features = min_features

    print(f"Saving processed graphs to {output_path}...")
    for name, g in tqdm(zip(filenames, graphs), total=len(graphs)):
        is_empty = (g.x.sum(dim=1) == 0).int().unsqueeze(1)
        g.x = torch.cat([g.x, is_empty], dim=1)
        g.num_node_features = g.x.shape[1]
        if hasattr(g, "feature_columns"):
            g.feature_columns = list(g.feature_columns) + ["is_empty"]
        save_path = output_path / name
        torch.save(g, save_path)

if __name__ == "__main__":
    main()
