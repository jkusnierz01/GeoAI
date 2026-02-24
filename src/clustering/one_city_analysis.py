import argparse
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN  # Added DBSCAN
from sklearn.preprocessing import normalize
import geopandas as gpd
import contextily as cx
from shapely.geometry import Point
import rootutils

ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.utils.model_utils import load_model_from_checkpoint

# --- CONFIGURATION ---
CITY = "warsaw"

# Clustering Algorithm Settings
CLUSTERING_ALGO = "dbscan"  # Options: 'kmeans', 'dbscan'

# KMeans Parameters
NUM_CLUSTERS = 12

# DBSCAN Parameters
# Note: Since embeddings are normalized, eps represents euclidean distance on the unit sphere.
# Start small (e.g., 0.1 - 0.5). If you get only -1 (noise), increase eps.
DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 10

# Filter Settings
WARSAW_CENTER_COORDS = (21.0122, 52.2297) # (Lon, Lat)
FILTER_RADIUS_METERS = 14000              # 14km

# Path Settings
GRAPH_PATH = ROOT / f"dataset_aligned/{CITY}_hexagons_res9.pt"
GEOJSON_PATH = ROOT / f"data/geodata/{CITY}_hexagons_res9.geojson"
MODEL_PATH = ROOT / "checkpoints/plain.ckpt"
CONFIG_PATH = ROOT / "configs/defaults.yaml"

# Fallback feature names
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

def get_all_node_embeddings(graph, model, device):
    """Generates embeddings for all nodes using the GNN."""
    x = graph.x.to(device)
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        graph.edge_index = graph.edge_index[:2]
        edge_index = graph.edge_index.to(device)
    else:
        edge_index = None

    with torch.no_grad():
        node_embeds = model.embed(x, edge_index)
    return node_embeds.cpu().numpy()

def analyze_and_save_clusters(embeddings, raw_features, feature_names, city_name, 
                              algo="kmeans", k=8, eps=0.5, min_samples=5):
    """
    Performs clustering (KMeans or DBSCAN) on the PROVIDED (filtered) embeddings.
    """
    print(f"--- Running {algo.upper()} Clustering on {len(embeddings)} nodes ---")
    
    # 1. Normalize
    embeds_norm = normalize(embeddings)
    
    # 2. Fit Algorithm
    if algo == "kmeans":
        print(f"   Params: k={k}")
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(embeds_norm)
    elif algo == "dbscan":
        print(f"   Params: eps={eps}, min_samples={min_samples}")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(embeds_norm)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # 3. Prepare Data for Interpretation
    if raw_features.shape[1] != len(feature_names):
        print(f"Warning: {raw_features.shape[1]} features in graph, but {len(feature_names)} names provided.")
        current_names = [f"feat_{i}" for i in range(raw_features.shape[1])]
    else:
        current_names = feature_names

    df = pd.DataFrame(raw_features, columns=current_names)
    df['cluster'] = labels
    
    # Get unique labels (DBSCAN might have -1, and variable number of clusters)
    unique_labels = sorted(df['cluster'].unique())
    n_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"--- Found {n_found} clusters (plus noise if -1 exists) ---")

    # 4. Calculate Lift (Cluster Mean / Global Mean)
    # We filter out noise (-1) for the global mean calculation to keep the baseline 'clean' (optional)
    # or keep it. Here we use the whole filtered dataset as baseline.
    global_means = df.drop(columns='cluster').mean() + 1e-6
    cluster_profiles = df.groupby('cluster').mean() + 1e-6
    lift_scores = cluster_profiles / global_means
    
    # 5. Interpretation Report
    report_filename = f"cluster_interpretation_{city_name}.txt"
    print(f"--- Saving interpretation to {report_filename} ---")
    
    cluster_names = {}
    report_lines = []
    report_lines.append(f"CLUSTER INTERPRETATION REPORT FOR {city_name.upper()}")
    report_lines.append(f"Algorithm: {algo.upper()}")
    report_lines.append("="*50)
    
    for cluster_id in unique_labels:
        # Handle DBSCAN Noise
        if cluster_id == -1:
            count = len(df[df['cluster'] == -1])
            report_lines.append(f"\nCluster -1 (Noise/Outliers): {count} nodes")
            report_lines.append("  (Nodes that did not fit into any dense cluster)")
            cluster_names[cluster_id] = "Noise"
            continue

        # Normal Clusters
        top_features = lift_scores.loc[cluster_id].sort_values(ascending=False).head(5)
        top_3_str = ", ".join([f"{idx}({val:.1f}x)" for idx, val in top_features.head(3).items()])
        
        count = len(df[df['cluster'] == cluster_id])
        report_lines.append(f"\nCluster {cluster_id} (n={count}):")
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

    # 2. Get Embeddings (FULL GRAPH)
    print(f"--- Generating embeddings for full graph ---")
    all_embeddings = get_all_node_embeddings(graph, model, device)
    all_raw_features = graph.x.cpu().numpy()
    
    # Ensure H3 IDs exist
    if not hasattr(graph, 'h3_ids'):
        print("Error: Graph object must have 'h3_ids' attribute.")
        return
    all_h3_ids = np.array([str(h) for h in graph.h3_ids])

    # 3. PRE-PROCESSING FILTER
    print(f"--- Loading GeoJSON for spatial filtering ---")
    gdf = gpd.read_file(args.geojson_path)
    
    # Prepare center point and project to meters (Web Mercator)
    center_point = Point(WARSAW_CENTER_COORDS)
    gdf_meters = gdf.to_crs(epsg=3857)
    center_proj = gpd.GeoSeries([center_point], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    
    # Calculate distance and find valid H3 IDs
    distances = gdf_meters.geometry.centroid.distance(center_proj)
    valid_gdf = gdf[distances <= FILTER_RADIUS_METERS]
    valid_h3_set = set(valid_gdf['h3_id'].astype(str))
    
    print(f"Spatial Filter: {len(gdf)} -> {len(valid_h3_set)} hexes (Radius: {FILTER_RADIUS_METERS/1000}km)")

    # 4. SLICE DATA
    keep_mask = np.array([h in valid_h3_set for h in all_h3_ids])
    
    if keep_mask.sum() == 0:
        print("Error: No graph nodes matched the spatial filter. Check coordinate systems or IDs.")
        return

    filtered_embeddings = all_embeddings[keep_mask]
    filtered_features = all_raw_features[keep_mask]
    filtered_h3_ids = all_h3_ids[keep_mask]
    
    print(f"Graph Filter: {len(all_embeddings)} -> {len(filtered_embeddings)} nodes used for clustering.")

    # 5. CLUSTERING (On filtered data only)
    # Passing the new config constants here
    cluster_labels, cluster_names = analyze_and_save_clusters(
        filtered_embeddings, 
        filtered_features, 
        feature_names, 
        CITY, 
        algo=CLUSTERING_ALGO,
        k=NUM_CLUSTERS,
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES
    )

    # 6. Save Cluster Assignments
    assignment_df = pd.DataFrame({
        'h3_index': filtered_h3_ids,
        'cluster_id': cluster_labels
    })
    assignment_df['cluster_meaning'] = assignment_df['cluster_id'].map(cluster_names)
    
    csv_filename = f"cluster_assignments_{CITY}.csv"
    assignment_df.to_csv(csv_filename, index=False)
    print(f"--- Saved filtered cluster assignments to {csv_filename} ---")

    # 7. Plotting
    print("--- Plotting Cluster Map ---")
    gdf_web = valid_gdf.to_crs(epsg=3857).copy()
    
    # Map IDs
    h3_to_cluster = dict(zip(assignment_df['h3_index'], assignment_df['cluster_id']))
    gdf_web['cluster_id'] = gdf_web['h3_id'].map(h3_to_cluster)
    gdf_web['cluster_label'] = gdf_web['cluster_id'].map(cluster_names)
    
    # Drop map rows that didn't get a cluster
    gdf_web = gdf_web.dropna(subset=['cluster_id'])

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    gdf_web.plot(
        column='cluster_label', 
        ax=ax, 
        alpha=0.6, 
        categorical=True, 
        legend=True, 
        cmap='tab20',
        edgecolor='none',
        legend_kwds={'bbox_to_anchor': (1.05, 1), 'loc': 'upper left', 'title': 'Dominant Features'}
    )
    
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    plt.title(f"Semantic Clusters ({CLUSTERING_ALGO.upper()}) - {CITY.capitalize()}")
    plt.tight_layout()
    
    # Dynamic filename based on algo
    filename_suffix = f"{NUM_CLUSTERS}" if CLUSTERING_ALGO == "kmeans" else f"eps{DBSCAN_EPS}"
    plt.savefig(f"cluster_map_{CITY}_{CLUSTERING_ALGO}_{filename_suffix}.png", dpi=300, bbox_inches="tight")
    print(f"--- Map saved ---")

if __name__ == "__main__":
    main()