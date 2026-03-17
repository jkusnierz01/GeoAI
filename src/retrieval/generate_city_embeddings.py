import os
import torch
import numpy as np
import rootutils
from pathlib import Path

# Set up project root
ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)
from src.utils.model_utils import load_model_from_checkpoint

# --- CONFIGURATION ---
MODEL_PATH = ROOT / "checkpoints/plain.ckpt"
CONFIG_PATH = ROOT / "configs/defaults.yaml"
DATA_DIR = ROOT / "dataset_aligned"
OUTPUT_DIR = ROOT / "data/vec2vec_inputs"  # Where we will save the .npy files

# The cities you want to align
CITIES = ["bogota", "paris", "madrid"]

def generate_and_save_embeddings(city_name, model, device, output_path):
    """
    Loads a city's graph, passes it through LEON to get 128D embeddings,
    and saves the raw vector pool as a .npy file for vec2vec.
    """
    graph_path = DATA_DIR / f"{city_name}_hexagons_res9.pt"
    
    if not graph_path.exists():
        print(f"Error: Could not find {graph_path}")
        return
        
    print(f"Loading {city_name} graph...")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    
    x = graph.x.to(device)
    edge_index = graph.edge_index[:2].to(device)
    
    print(f"Generating 128-dimensional embeddings for {city_name}...")
    with torch.no_grad():
        # Extracted embeddings will have shape (num_hexagons, 128) [cite: 113, 172]
        node_embeds = model.embed(x, edge_index).cpu().numpy()
        
    # Optional but recommended for vec2vec: L2 Normalize the embeddings
    # This maps them to a unit hypersphere, making angular alignment easier
    norms = np.linalg.norm(node_embeds, axis=1, keepdims=True)
    node_embeds_normalized = node_embeds / (norms + 1e-9)
    
    # Save the raw array
    np.save(output_path, node_embeds_normalized)
    print(f"Saved {city_name} embeddings with shape {node_embeds_normalized.shape} to {output_path}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize LEON Model
    # Assuming the input features N=32 [cite: 117] and hidden state is 128 
    print("Loading LEON model...")
    model = load_model_from_checkpoint(MODEL_PATH, 32, CONFIG_PATH).to(device)
    model.eval()
    
    # 2. Process each city
    for city in CITIES:
        output_file = OUTPUT_DIR / f"{city}_embeds_128d.npy"
        generate_and_save_embeddings(city, model, device, output_file)
        
    print("Data preparation complete! You are ready for vec2vec.")

if __name__ == "__main__":
    main()