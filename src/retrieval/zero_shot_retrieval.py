import torch
import numpy as np
import pandas as pd
import h3
import rootutils
from sklearn.metrics.pairwise import cosine_similarity

ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)
from src.utils.model_utils import load_model_from_checkpoint

# --- CONFIGURATION ---
MODEL_PATH = ROOT / "checkpoints/plain.ckpt"
CONFIG_PATH = ROOT / "configs/defaults.yaml"
DATA_DIR = ROOT / "dataset_aligned"

# Cities to compare
QUERY_CITY ="tokyo"
TARGET_CITY = "warsaw" 

RESOLUTION = 9  # H3 resolution used in your dataset
INDEX = 40074
TOP_K = 3

# Provide the FALLBACK_FEATURE_NAMES from your previous script
FEATURE_NAMES = [
    'amenity_hospital', 'amenity_pharmacy', 'amenity_bank', 'amenity_police', 
    'shop_supermarket', 'shop_bakery', 'shop_greengrocer', 'shop_alcohol', 
    'shop_clothes', 'amenity_restaurant', 'amenity_bar', 'amenity_nightclub', 
    'tourism_hotel', 'tourism_museum', 'landuse_cemetery', 'landuse_industrial', 
    'leisure_park', 'leisure_sports_centre', 'leisure_playground', 'building_office', 
    'building_house', 'building_apartments', 'railway_station', 'railway_tram_stop', 
    'highway_bus_stop', 'aeroway_aerodrome', 'amenity_cinema', 'amenity_theatre', 
    'amenity_library', 'amenity_place_of_worship', 'amenity_school', 'is_empty'
]

def load_city_data(city_name, model, device):
    """Loads graph, generates embeddings, and extracts H3/Features."""
    path = DATA_DIR / f"{city_name}_hexagons_res{RESOLUTION}.pt"
    graph = torch.load(path, map_location="cpu", weights_only=False)
    
    # Generate Embeddings
    x = graph.x.to(device)
    edge_index = graph.edge_index[:2].to(device)
    with torch.no_grad():
        embeds = model.embed(x, edge_index).cpu().numpy()
    
    # Normalize for Cosine Similarity
    embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)
    
    # Get H3 IDs (assumes they are stored as strings or ints in graph.h3_ids)
    h3_ids = [str(hid) for hid in graph.h3_ids]
    features = graph.x.cpu().numpy()
    
    return embeds, h3_ids, features

def main(query_idx=100, top_k=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize LEON Model
    # Note: Using 32 as the input feature dimension from your paper [cite: 117]
    model = load_model_from_checkpoint(MODEL_PATH, 32, CONFIG_PATH).to(device)
    model.eval()

    print(f"--- Loading {QUERY_CITY} and {TARGET_CITY} ---")
    q_embeds, q_h3, q_feats = load_city_data(QUERY_CITY, model, device)
    t_embeds, t_h3, t_feats = load_city_data(TARGET_CITY, model, device)

    # 1. Perform Retrieval
    query_vec = q_embeds[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_vec, t_embeds).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 2. Get Query Info
    q_h3_val = q_h3[query_idx]
    q_coords = h3.cell_to_latlng(q_h3_val)
    print(f"\nSOURCE: {QUERY_CITY} Index {query_idx}")
    print(f"H3: {q_h3_val} | Coords: {q_coords}")
    print(f"Google Maps: https://www.google.com/maps?q={q_coords[0]},{q_coords[1]}")
    print("="*80)

    # 3. Compare with Top Matches
    for i, t_idx in enumerate(top_indices):
        t_h3_val = t_h3[t_idx]
        t_coords = h3.cell_to_latlng(t_h3_val)
        score = similarities[t_idx]
        
        print(f"\nMATCH #{i+1} (Score: {score:.4f})")
        print(f"City: {TARGET_CITY} | Target Index: {t_idx}")
        print(f"H3: {t_h3_val} | Coords: {t_coords}")
        print(f"Google Maps: https://www.google.com/maps?q={t_coords[0]},{t_coords[1]}")
        
        # 4. Feature Comparison Table 
        print(f"{'AMENITY':<25} | {'QUERY (WAW)':<12} | {'MATCH (TKY)':<12} | {'DIFF'}")
        print("-" * 65)
        
        for f_i, name in enumerate(FEATURE_NAMES):
            q_val = q_feats[query_idx][f_i]
            m_val = t_feats[t_idx][f_i]
            
            # Show only features present in either the query or the match
            if q_val > 0 or m_val > 0:
                print(f"{name:<25} | {q_val:<12.1f} | {m_val:<12.1f} | {m_val-q_val:+.1f}")
        print("-" * 80)

if __name__ == "__main__":
    # Change query_idx to any index you want to investigate from Warsaw
    main(query_idx=INDEX, top_k=TOP_K)