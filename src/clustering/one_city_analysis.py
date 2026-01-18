import argparse
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import geopandas as gpd
import contextily as cx
import rootutils

ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.utils.model_utils import load_model_from_checkpoint

# --- CONFIGURATION ---
CITY = "warsaw"

# Fallback feature names found in your specific file
FALLBACK_FEATURE_NAMES = [
    'amenity_hospital', 'amenity_pharmacy', 'amenity_bank', 'amenity_police', 
    'shop_supermarket', 'shop_bakery', 'shop_greengrocer', 'shop_alcohol', 
    'shop_clothes', 'amenity_restaurant', 'amenity_bar', 'amenity_nightclub', 
    'tourism_hotel', 'tourism_museum', 'landuse_cemetery', 'landuse_industrial', 
    'leisure_park', 'leisure_sports_centre', 'leisure_playground', 'building_office', 
    'building_house', 'building_apartments', 'railway_station', 'railway_tram_stop', 
    'highway_bus_stop', 'aeroway_aerodrome', 'amenity_cinema', 'amenity_theatre', 
    'amenity_library', 'amenity_place_of_worship', 'amenity_school', 'is_empty'
]

GRAPH_PATH = ROOT / f"dataset_aligned/{CITY}_hexagons_res9.pt"
GEOJSON_PATH = ROOT / f"data/geodata/{CITY}_hexagons_res9.geojson"
MODEL_PATH = ROOT / "checkpoints/plain.ckpt"
CONFIG_PATH = ROOT / "configs/defaults.yaml"
NUM_CLUSTERS = 6

def get_all_node_embeddings(graph, model, device):
    """Generates embeddings for all nodes."""
    x = graph.x.to(device)
    graph.edge_index = graph.edge_index[:2]
    edge_index = graph.edge_index.to(device)
    with torch.no_grad():
        node_embeds = model.embed(x, edge_index)
    return node_embeds.cpu().numpy()

def analyze_and_save_clusters(embeddings, raw_features, feature_names, city_name, k=8):
    """
    Performs clustering, calculates interpretations, and saves results to files.
    """
    print(f"--- Running K-Means Clustering (k={k}) ---")
    
    # 1. Normalize and Cluster
    embeds_norm = normalize(embeddings)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeds_norm)
    
    # 2. Prepare Data for Interpretation
    if raw_features.shape[1] != len(feature_names):
        print(f"Warning: {raw_features.shape[1]} features in graph, but {len(feature_names)} names provided.")
        current_names = [f"feat_{i}" for i in range(raw_features.shape[1])]
    else:
        current_names = feature_names

    df = pd.DataFrame(raw_features, columns=current_names)
    df['cluster'] = labels
    
    # 3. Calculate Lift (Cluster Mean / Global Mean)
    global_means = df.drop(columns='cluster').mean() + 1e-6
    cluster_profiles = df.groupby('cluster').mean() + 1e-6
    lift_scores = cluster_profiles / global_means
    
    # 4. Generate and Save Interpretation Report
    report_filename = f"cluster_interpretation_{city_name}.txt"
    print(f"--- Saving interpretation to {report_filename} ---")
    
    cluster_names = {}
    report_lines = []
    report_lines.append(f"CLUSTER INTERPRETATION REPORT FOR {city_name.upper()}")
    report_lines.append("="*50)
    
    for cluster_id in range(k):
        # Get top 5 features for detailed report
        top_features = lift_scores.loc[cluster_id].sort_values(ascending=False).head(5)
        
        # Format for console/dictionary (short)
        top_3_str = ", ".join([f"{idx}({val:.1f}x)" for idx, val in top_features.head(3).items()])
        
        # Format for text file (detailed)
        report_lines.append(f"\nCluster {cluster_id}:")
        report_lines.append(f"  Dominant Features (vs City Avg):")
        for idx, val in top_features.items():
            report_lines.append(f"    - {idx}: {val:.2f}x higher")
            
        print(f"Cluster {cluster_id}: {top_3_str}")
        
        dominant_feature = top_features.index[0]
        cluster_names[cluster_id] = f"{cluster_id}: {dominant_feature}"

    with open(report_filename, "w") as f:
        f.write("\n".join(report_lines))
        
    return labels, cluster_names

def main():
    parser = argparse.ArgumentParser(description="Visualize and save node clusters.")
    parser.add_argument("--graph_path", default=GRAPH_PATH)
    parser.add_argument("--geojson_path", default=GEOJSON_PATH)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--config_path", default=CONFIG_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"--- Loading graph: {args.graph_path} ---")
    graph = torch.load(args.graph_path, map_location="cpu", weights_only=False)
    
    if hasattr(graph, 'feature_names'):
        feature_names = graph.feature_names
    else:
        feature_names = FALLBACK_FEATURE_NAMES

    model = load_model_from_checkpoint(args.model_path, graph.num_node_features, args.config_path).to(device)

    # 2. Get Embeddings
    print(f"--- Generating embeddings ---")
    embeddings = get_all_node_embeddings(graph, model, device)
    
    # 3. CLUSTERING & SAVING INTERPRETATIONS
    raw_features = graph.x.cpu().numpy()
    cluster_labels, cluster_names = analyze_and_save_clusters(embeddings, raw_features, feature_names, CITY, k=NUM_CLUSTERS)

    # 4. Save Cluster Assignments (Hex ID -> Cluster ID)
    if hasattr(graph, 'h3_ids'):
        h3_ids = graph.h3_ids
    else:
        print("Error: Graph is missing 'h3_ids'.")
        return

    # Create a DataFrame for the CSV
    assignment_df = pd.DataFrame({
        'h3_index': [str(h) for h in h3_ids],
        'cluster_id': cluster_labels
    })
    
    # Add the "Cluster Name" (e.g., "0: amenity_restaurant") for easier reading
    assignment_df['cluster_meaning'] = assignment_df['cluster_id'].map(cluster_names)
    
    csv_filename = f"cluster_assignments_{CITY}.csv"
    assignment_df.to_csv(csv_filename, index=False)
    print(f"--- Saved cluster assignments to {csv_filename} ---")

# 5. Plotting (Corrected Legend)
    print(f"--- Loading GeoJSON ---")
    gdf = gpd.read_file(args.geojson_path)
    
    # Map H3 IDs to Cluster IDs
    h3_to_cluster = dict(zip(assignment_df['h3_index'], assignment_df['cluster_id']))
    gdf['cluster_id'] = gdf['h3_id'].map(h3_to_cluster)
    
    # ---------------------------------------------------------
    # FIX: Map Cluster IDs to meaningful Label strings for the Legend
    # ---------------------------------------------------------
    gdf['cluster_label'] = gdf['cluster_id'].map(cluster_names)
    
    print("--- Plotting Cluster Map ---")
    gdf_web = gdf.to_crs(epsg=3857).dropna(subset=['cluster_id'])
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Plot using 'cluster_label' instead of 'cluster_id'
    gdf_web.plot(
        column='cluster_label',  # Use the meaningful text column
        ax=ax, 
        alpha=0.6, 
        categorical=True, 
        legend=True, 
        cmap='tab20',            # Using tab20 because you have ~12 clusters
        edgecolor='none',
        # Move legend outside the map so it doesn't block the city
        legend_kwds={
            'bbox_to_anchor': (1.05, 1), 
            'loc': 'upper left', 
            'title': 'Dominant Features'
        }
    )
    
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    
    plt.title(f"Semantic Clusters - {CITY.capitalize()}")
    plt.tight_layout() # Adjust layout to make room for legend
    plt.savefig(f"cluster_map_{CITY}_{NUM_CLUSTERS}.png", dpi=300, bbox_inches="tight")
    print(f"--- Map saved to cluster_map_{CITY}_{NUM_CLUSTERS}.png ---")

if __name__ == "__main__":
    main()