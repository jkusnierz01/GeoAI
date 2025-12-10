import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import geopandas as gpd
from types import SimpleNamespace
from omegaconf import OmegaConf
from graphmae.models import build_model
import rootutils
import contextily as cx
from utils.model_utils import load_model_from_checkpoint

ROOT = rootutils.setup_root(search_from=".", indicator=".project_root", pythonpath=True)
GRAPH_PATH = ROOT / "dataset_aligned/berlin_hexagons_res8.pt"
GEOJSON_PATH = ROOT / "geodata/berlin_hexagons_res8.geojson"
MODEL_PATH = ROOT / "outputs/2025-11-22/21-20-43/checkpoint.pt"
CONFIG_PATH = ROOT / "outputs/2025-11-22/21-20-43/.hydra/config.yaml"

def get_all_node_embeddings(graph, model, device):
    """Generuje embeddingi dla wszystkich węzłów w danym grafie."""
    x = graph.x.to(device)
    graph.edge_index = graph.edge_index[:2]
    edge_index = graph.edge_index.to(device)
    with torch.no_grad():
        node_embeds = model.embed(x, edge_index)
    return node_embeds.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Visualize node embeddings on a map.")
    parser.add_argument("--graph_path", type=str, required=False, help="Path to the graph .pt file for a single city.", default=GRAPH_PATH)
    parser.add_argument("--geojson_path", type=str, required=False, help="Path to the corresponding .geojson file for plotting.", default=GEOJSON_PATH)
    parser.add_argument("--model_path", type=str, required=False, help="Path to the trained model checkpoint.pt.", default=MODEL_PATH)
    parser.add_argument("--config_path", type=str, required=False, help="Path to the defaults.yaml config file used for training.", default=CONFIG_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"--- Loading graph: {args.graph_path} ---")
    graph = torch.load(args.graph_path, map_location="cpu", weights_only=False)
    
    if not hasattr(graph, 'h3_ids'):
        print("Error: The loaded graph object does not have the 'h3_ids' attribute.")
        return

    model = load_model_from_checkpoint(args.model_path, graph.num_node_features, args.config_path).to(device)

    print(f"--- Generating embeddings for all {graph.num_nodes} nodes ---")
    embeddings = get_all_node_embeddings(graph, model, device)

    print("--- Running PCA to reduce embedding dimensions to 3 for RGB colors ---")
    pca = PCA(n_components=3)
    embeds_3d = pca.fit_transform(embeddings)
    embeds_norm = (embeds_3d - embeds_3d.min(0)) / (embeds_3d.max(0) - embeds_3d.min(0))
    print(f"Shape of normalized colors: {embeds_norm.shape}")

    print(f"--- Loading GeoJSON for plotting: {args.geojson_path} ---")
    gdf = gpd.read_file(args.geojson_path)
    
    h3_to_color = {h3_id: color for h3_id, color in zip(graph.h3_ids, embeds_norm)}
    gdf['plot_color'] = gdf['h3_id'].apply(lambda h3: h3_to_color.get(h3, (0, 0, 0)))

    print("--- Plotting the semantic map with basemap ---")
    
    # --- ZMIANA 1: Konwersja systemu współrzędnych ---
    # Mapy bazowe używają systemu Web Mercator (EPSG:3857)
    print("Converting GeoDataFrame to Web Mercator projection (EPSG:3857)...")
    gdf_web_mercator = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # --- ZMIANA 2: Rysowanie heksagonów z przezroczystością ---
    # Używamy alpha, aby podkład mapy był widoczny 
    gdf_web_mercator.plot(color=gdf_web_mercator['plot_color'].tolist(), ax=ax, alpha=0.6, edgecolor='none')

    # --- ZMIANA 3: Dodanie mapy bazowej ---
    # contextily automatycznie pobierze kafelki dla widocznego obszaru
    print("Adding basemap from contextily...")
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)

    city_name = os.path.basename(args.graph_path).split('_hexagons')[0]
    ax.set_xticks([])
    ax.set_yticks([])
    
    output_filename = f"semantic_map_with_basemap_{city_name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"--- Map saved to: {output_filename} ---")

if __name__ == "__main__":
    main()