import argparse
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import normalize
import geopandas as gpd
import contextily as cx
import rootutils

# Setup root
ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.utils.model_utils import load_model_from_checkpoint

# --- CONFIGURATION ---
# Select 3 distinct cities to show generalization
CITIES = ["sydney", "hanoi", "berlin"] 

# Path configuration
MODEL_PATH = ROOT / "checkpoints/plain.ckpt"
CONFIG_PATH = ROOT / "configs/defaults.yaml"
KMEANS_PATH = ROOT / "checkpoints/kmeans_global_model.joblib"  # Make sure this exists!

# --- CORRECTED LEGEND BASED ON K=12 DATA ---
CLUSTER_SEMANTICS = {
    0: "Low-Density Res. (Houses)",
    1: "Public Services (Schools/Gov)",
    2: "Religious & Institutional",
    3: "CBD (Offices)",
    4: "Local Commercial & Services",
    5: "Green Space (Parks)",
    6: "Industrial & Logistics",
    7: "Tourism & Entertainment",
    8: "Transit Corridors",
    9: "Undeveloped / Empty",
    10: "Sports & Culture",
    11: "High-Density Res. (Apts)"
}

def get_node_embeddings(city, model, device):
    """Loads graph and generates embeddings for a single city."""
    graph_path = ROOT / f"dataset_aligned/{city}_hexagons_res9.pt"
    if not graph_path.exists():
        print(f"Skipping {city}: File not found.")
        return None, None

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    x = graph.x.to(device)
    
    # Handle edge index (ignore weights if present)
    graph.edge_index = graph.edge_index[:2]
    edge_index = graph.edge_index.to(device)
    
    with torch.no_grad():
        embeddings = model.embed(x, edge_index)
        
    return embeddings.cpu().numpy(), graph.h3_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--kmeans_path", default=KMEANS_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Global Models
    print(f"Loading GNN from {args.model_path}...")
    # Load first city just to get feature dims
    ref_graph = torch.load(ROOT / f"dataset_aligned/{CITIES[0]}_hexagons_res9.pt", weights_only=False)
    model = load_model_from_checkpoint(args.model_path, ref_graph.num_node_features, CONFIG_PATH).to(device)
    
    print(f"Loading Global K-Means from {args.kmeans_path}...")
    try:
        kmeans = joblib.load(args.kmeans_path)
    except FileNotFoundError:
        print("ERROR: Global K-Means model not found. Run your global clustering script first!")
        return

    # 2. Setup Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    axes = axes.flatten()

    # Use 'tab20' which has 20 distinctive colors. 
    # We will map our 12 clusters to the first 12 colors consistently.
    cmap = plt.get_cmap("tab20")
    
    # 3. Process Each City
    for i, city in enumerate(CITIES):
        ax = axes[i]
        print(f"Processing {city}...")
        
        # A. Get Embeddings
        embeddings, h3_ids = get_node_embeddings(city, model, device)
        if embeddings is None:
            continue
            
        # B. Predict Clusters (Using GLOBAL model)
        embeds_norm = normalize(embeddings)
        labels = kmeans.predict(embeds_norm)
        
        # C. Prepare GeoData
        geojson_path = ROOT / f"data/geodata/{city}_hexagons_res9.geojson"
        gdf = gpd.read_file(geojson_path)
        
        # Map IDs to Clusters
        h3_to_cluster = dict(zip([str(h) for h in h3_ids], labels))
        gdf['cluster_id'] = gdf['h3_id'].map(h3_to_cluster)
        
        # D. Plot
        gdf_plot = gdf.dropna(subset=['cluster_id']).to_crs(epsg=3857)
        
        gdf_plot.plot(
            column='cluster_id',
            ax=ax,
            cmap=cmap,
            categorical=True,
            alpha=0.7,
            edgecolor='none',
            # Important: Fix the color range so Cluster 0 is always the same color
            vmin=0, 
            vmax=19 
        )
        
        # Add Basemap
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, attribution=False)
        ax.set_axis_off()
        ax.set_title(city.replace("_", " ").title(), fontsize=20, fontweight='bold')

    # 4. Create Unified Legend
    print("Creating unified legend...")
    
    legend_handles = []
    # Loop through our valid cluster IDs (0 to 11)
    for cid in sorted(CLUSTER_SEMANTICS.keys()):
        label = CLUSTER_SEMANTICS[cid]
        color = cmap(cid) # Use same colormap index
        patch = mpatches.Patch(color=color, label=f"{cid}: {label}")
        legend_handles.append(patch)

    # Place legend at bottom
    fig.legend(
        handles=legend_handles,
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,  # 4 columns x 3 rows = 12 items
        fontsize=14,
        frameon=True,
        title="Universal Urban Functional Zones (K=12)"
    )

    plt.subplots_adjust(bottom=0.20, wspace=0.05)
    
    save_path = f"universal_clusters_K12.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Success! Figure saved to {save_path}")

if __name__ == "__main__":
    main()