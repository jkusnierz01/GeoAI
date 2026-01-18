import argparse
import os
import glob
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import rootutils
import joblib

ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

from src.utils.model_utils import load_model_from_checkpoint

# --- CONFIGURATION ---
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

MODEL_PATH = ROOT / "checkpoints/plain.ckpt"
CONFIG_PATH = ROOT / "configs/defaults.yaml"
DATA_DIR = ROOT / "dataset_aligned"
# We define 16 global clusters to capture diversity across multiple cities
NUM_CLUSTERS = 12

def get_all_cities():
    """Scans the dataset folder for all available city graphs."""
    pattern = str(DATA_DIR / "*_hexagons_res9.pt")
    files = glob.glob(pattern)
    cities = [os.path.basename(f).replace("_hexagons_res9.pt", "") for f in files]
    return sorted(cities)

def get_all_node_embeddings(graph, model, device):
    """Generates embeddings for all nodes."""
    x = graph.x.to(device)
    graph.edge_index = graph.edge_index[:2]
    edge_index = graph.edge_index.to(device)
    with torch.no_grad():
        node_embeds = model.embed(x, edge_index)
    return node_embeds.cpu().numpy()

def analyze_global_clusters(all_embeddings, all_features, feature_names, k=8):
    """
    Performs clustering on the COMBINED dataset and prints interpretation.
    """
    print(f"\n>>> RUNNING GLOBAL CLUSTERING ON {len(all_embeddings)} HEXAGONS (k={k}) <<<")
    
    # [cite_start]1. Normalize (Cosine Similarity proxy [cite: 186])
    print("Normalizing embeddings...")
    embeds_norm = normalize(all_embeddings)
    
    # 2. Global Clustering
    print("Fitting K-Means...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeds_norm)
    joblib.dump(kmeans, ROOT / "kmeans_global_model.joblib")
    
    # 3. Prepare Data for Interpretation
    if all_features.shape[1] != len(feature_names):
        print(f"Warning: {all_features.shape[1]} features in data, but {len(feature_names)} names provided.")
        current_names = [f"feat_{i}" for i in range(all_features.shape[1])]
    else:
        current_names = feature_names

    print("Calculating Feature Lift...")
    df = pd.DataFrame(all_features, columns=current_names)
    df['cluster'] = labels
    
    # 4. Calculate Lift (Cluster Mean / Global Mean)
    # Using epsilon to avoid division by zero
    global_means = df.drop(columns='cluster').mean() + 1e-6
    cluster_profiles = df.groupby('cluster').mean() + 1e-6
    lift_scores = cluster_profiles / global_means
    
    print("\n" + "="*80)
    print(f"{'CLUSTER ID':<12} | {'DOMINANT FEATURES (Lift vs Global Avg)':<60}")
    print("="*80)
    
    for cluster_id in range(k):
        # Get top 4 features for detailed view
        top_features = lift_scores.loc[cluster_id].sort_values(ascending=False).head(4)
        
        # Format string
        features_str = ", ".join([f"{name} ({val:.1f}x)" for name, val in top_features.items()])
        print(f"Cluster {cluster_id:02d}     | {features_str}")
    print("="*80 + "\n")
        
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--config_path", default=CONFIG_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Find Cities
    cities = get_all_cities()
    print(f"Found {len(cities)} cities: {cities}")
    
    if not cities:
        print("No .pt files found.")
        return

    # 2. Initialize Model
    # Load first graph to determine input dimension
    first_path = DATA_DIR / f"{cities[0]}_hexagons_res9.pt"
    first_graph = torch.load(first_path, map_location="cpu", weights_only=False)
    
    # Determine feature names (assume consistent across all cities)
    if hasattr(first_graph, 'feature_names'):
        global_feature_names = first_graph.feature_names
    else:
        global_feature_names = FALLBACK_FEATURE_NAMES
        
    model = load_model_from_checkpoint(args.model_path, first_graph.num_node_features, args.config_path).to(device)

    # 3. Accumulate Data
    all_embeddings_list = []
    all_features_list = []
    total_hexagons = 0

    print("\n--- Accumulating data from all cities ---")
    for city in cities:
        try:
            graph_path = DATA_DIR / f"{city}_hexagons_res9.pt"
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            
            # Generate Embeddings
            emb = get_all_node_embeddings(graph, model, device)
            feat = graph.x.cpu().numpy()
            
            all_embeddings_list.append(emb)
            all_features_list.append(feat)
            
            total_hexagons += emb.shape[0]
            print(f"Loaded {city}: {emb.shape[0]} hexagons")
            
        except Exception as e:
            print(f"Error loading {city}: {e}")

    # 4. Stack Everything
    if not all_embeddings_list:
        print("No data loaded.")
        return

    print(f"\nStacking {total_hexagons} total hexagons...")
    global_embeddings = np.vstack(all_embeddings_list)
    global_features = np.vstack(all_features_list)

    # 5. Run Global Analysis
    analyze_global_clusters(
        global_embeddings, 
        global_features, 
        global_feature_names, 
        k=NUM_CLUSTERS
    )

if __name__ == "__main__":
    main()