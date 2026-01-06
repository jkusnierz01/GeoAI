import os
import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import rootutils
import h3

# Setup root
ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.utils.model_utils import load_model_from_checkpoint

# ==========================================
# CONFIGURATION
# ==========================================
# POINT THIS TO YOUR PROCESSED/ALIGNED DATASET
INPUT_DIR = "dataset_aligned"  # Folder with PROCESSED graphs
MODEL_PATH = "checkpoints/plain.ckpt"
CONFIG_PATH = os.path.join(ROOT, "configs", "defaults.yaml")
OUTPUT_PLOT = "correlation_dist_emb_plot.png"
SAMPLES_PER_GRAPH = 1000

def get_lat_lon(h3_id):
    """Robust decoder with explicit error reporting."""
    try:
        if hasattr(h3_id, 'item'): h3_id = h3_id.item()
        h3_hex = hex(h3_id)[2:] if isinstance(h3_id, int) else h3_id
        
        if hasattr(h3, 'cell_to_latlng'):
            return h3.cell_to_latlng(h3_hex)
        elif hasattr(h3, 'h3_to_geo'):
            return h3.h3_to_geo(h3_hex)
    except Exception:
        return (0.0, 0.0)
    return (0.0, 0.0)

def cosine_sim(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    if norma == 0 or normb == 0: return 0.0
    return dot / (norma * normb)

def prepare_data_simple(data):
    """
    For PROCESSED files:
    1. Get Coords from H3.
    2. Returns x directly (already preprocessed).
    """
    x = data.x.float()
    
    # --- 1. Coordinates ---
    if not hasattr(data, 'h3_ids'):
        return None, None
        
    coords_list = []
    for hid in data.h3_ids:
        c = get_lat_lon(hid)
        coords_list.append(c)
        
    coords = torch.tensor(coords_list, dtype=torch.float)

    # --- 2. Features ---
    # No Log1p or column removal needed!
    return coords, x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar])
    except:
        pass

    # Load Model
    print(f"Loading model from {args.model_path}...")
    try:
        # Assuming processed graphs are size 32. If aligned to 33, change to 33.
        model = load_model_from_checkpoint(args.model_path, 32, args.config_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    files = glob.glob(os.path.join(args.input_dir, "*.pt"))
    print(f"Found {len(files)} processed files.")
    
    all_distances = []
    all_similarities = []
    processed_count = 0
    
    for file_path in tqdm(files):
        try:
            data = torch.load(file_path, weights_only=False)
            
            # Use simple preparation
            coords, x = prepare_data_simple(data)
            
            if coords is None: continue
            
            # Ensure dimensions match model (pad if needed)
            if x.shape[1] < 32:
                pad = torch.zeros((x.shape[0], 32 - x.shape[1]), device=x.device)
                x = torch.cat([x, pad], dim=1)
            elif x.shape[1] > 32:
                x = x[:, :32]
            
            x = x.to(device)
            
            # Slice edge index if needed
            edge_index = data.edge_index[:2].to(device)
            
            # Embed
            with torch.no_grad():
                if hasattr(model, 'embed'):
                    embeddings = model.embed(x, edge_index)
                elif hasattr(model, 'encoder'):
                    embeddings = model.encoder(x, edge_index)
                else:
                    continue
            
            embeddings = embeddings.cpu().numpy()
            coords = coords.numpy()
            
            # Sample pairs
            num_nodes = x.shape[0]
            if num_nodes < 2: continue

            processed_count += 1
            indices_a = np.random.choice(num_nodes, SAMPLES_PER_GRAPH)
            indices_b = np.random.choice(num_nodes, SAMPLES_PER_GRAPH)
            
            for idx_a, idx_b in zip(indices_a, indices_b):
                if idx_a == idx_b: continue
                
                dist = np.linalg.norm(coords[idx_a] - coords[idx_b])
                
                if dist < 1e-6: continue
                    
                sim = cosine_sim(embeddings[idx_a], embeddings[idx_b])
                all_distances.append(dist)
                all_similarities.append(sim)

        except Exception as e:
            # print(f"Error {os.path.basename(file_path)}: {e}")
            pass

    if not all_distances:
        print("\nFAILURE: No valid pairs generated.")
        return

    # Stats & Plot
    all_d = np.array(all_distances)
    all_s = np.array(all_similarities)
    
    print("\n" + "="*40)
    print(f"Graphs Processed: {processed_count}")
    print(f"Total Pairs: {len(all_d)}")
    print(f"Pearson:  {pearsonr(all_d, all_s)[0]:.4f}")
    print(f"Spearman: {spearmanr(all_d, all_s)[0]:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # 1. Hexbin
    plt.subplot(1, 2, 1)
    hb = plt.hexbin(all_d, all_s, gridsize=50, cmap='inferno', mincnt=1, bins='log')
    plt.colorbar(hb, label='Log Count')
    plt.title("Density: Distance vs Similarity")
    plt.xlabel("Geographic Distance (deg)")
    plt.ylabel("Cosine Similarity")

    # 2. Trend
    plt.subplot(1, 2, 2)
    bins = np.linspace(min(all_d), max(all_d), 20)
    bin_indices = np.digitize(all_d, bins)
    bin_means = []
    bin_centers = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.any(mask):
            bin_means.append(np.mean(all_s[mask]))
            bin_centers.append(0.5 * (bins[i] + bins[i-1]))
            
    plt.plot(bin_centers, bin_means, marker='o', color='blue')
    plt.title("Trend: Avg Similarity vs Distance")
    plt.xlabel("Geographic Distance (deg)")
    plt.ylabel("Average Cosine Similarity")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()