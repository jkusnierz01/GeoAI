import torch
import h3
import numpy as np
import rootutils

ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

# --- CONFIGURATION ---
CITY = "tokyo"
RESOLUTION = 9  # Based on your filenames 
DATA_DIR = ROOT / "dataset_aligned"
LAT, LNG = 35.6369546,139.7456833

def find_index_from_coords(city_name, lat, lng, res=9):
    """
    Converts Lat/Lng to H3 index and finds its position in the city graph.
    """
    # 1. Convert coordinate to H3 Index [cite: 78, 94]
    target_h3 = h3.latlng_to_cell(lat, lng, res)
    print(f"Target H3 Index for ({lat}, {lng}): {target_h3}")

    # 2. Load the city graph
    graph_path = DATA_DIR / f"{city_name}_hexagons_res{res}.pt"
    if not graph_path.exists():
        print(f"Error: Graph file for {city_name} not found.")
        return None

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    
    # 3. Search for the H3 ID in the graph's ID list
    if not hasattr(graph, 'h3_ids'):
        print("Error: Graph object lacks 'h3_ids' attribute.")
        return None

    # Handle both string and integer H3 IDs [cite: 76]
    all_ids = [str(hid) for hid in graph.h3_ids]
    
    try:
        idx = all_ids.index(target_h3)
        print(f"SUCCESS: H3 {target_h3} found at Index {idx} in {city_name}.")
        return idx
    except ValueError:
        print(f"NOT FOUND: H3 {target_h3} is not in the {city_name} graph.")
        print("Note: The location might be outside the urban boundaries used in training.")
        return None

if __name__ == "__main__":    
    idx = find_index_from_coords(CITY, LAT, LNG, res=RESOLUTION)
    
    if idx is not None:
        print(f"You can now use index {idx} in your retrieval or visualization scripts.")