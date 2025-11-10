"""
hexes_to_graph.py

Converts a GeoJSON file (from Overpass H3 downloader) into a PyTorch Geometric graph.

Input:
    A GeoJSON file created by your previous downloader script (one resolution),
    containing features with properties like:
        h3_id, resolution, <amenity keys...>, total_count

Output:
    A PyTorch Geometric Data object saved to disk as: graph_res{res}.pt

Example usage:
    python3 hexes_to_graph.py --input riyadh_hexagons_res8.geojson --out graph_res8.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import h3
import torch
from torch_geometric.data import Data


# ----------------------------
# Helpers
# ----------------------------

def load_geojson_to_gdf(path: str) -> gpd.GeoDataFrame:
    """Load GeoJSON as GeoDataFrame and ensure h3_id column exists."""
    gdf = gpd.read_file(path)
    if 'h3_id' not in gdf.columns:
        raise ValueError("GeoJSON must contain 'h3_id' property.")
    gdf['h3_id'] = gdf['h3_id'].astype(str)
    return gdf


def extract_amenity_columns(gdf: gpd.GeoDataFrame) -> list:
    """Return all numeric feature columns except reserved ones."""
    reserved = {'h3_id', 'geometry', 'resolution', 'total_count'}
    return sorted([
        c for c in gdf.columns
        if c not in reserved and gdf[c].dtype.kind in 'fi'
    ])


def compute_centroids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Compute centroid coordinates (x, y) in projected CRS."""
    # Project to meters to avoid geographic coordinate distortion
    gdf_proj = gdf.to_crs(epsg=3857)
    cents = gdf_proj.geometry.centroid
    return np.array([[pt.x, pt.y] for pt in cents])


def build_h3_adjacency(h3_ids: list[str]) -> np.ndarray:
    """Compute adjacency edges from H3 hex neighbors."""
    id_to_idx = {h: i for i, h in enumerate(h3_ids)}
    edges = set()
    h3_set = set(h3_ids)

    for h in h3_ids:
        i = id_to_idx[h]
        for nb in h3.k_ring(h, 1):
            if nb != h and nb in h3_set:
                j = id_to_idx[nb]
                edges.add((min(i, j), max(i, j)))

    if not edges:
        return np.empty((2, 0), dtype=int)

    edge_list = np.array(list(edges), dtype=int)
    edge_index = np.vstack([edge_list.T, edge_list.T[::-1]])  # bidirectional edges
    return edge_index


# ----------------------------
# Core
# ----------------------------

def build_graph_from_geojson(path_geojson: str, include_coords: bool = False) -> Data:
    """Build a PyTorch Geometric graph from an H3 GeoJSON."""
    gdf = load_geojson_to_gdf(path_geojson)
    gdf = gdf.sort_values('h3_id').reset_index(drop=True)
    h3_ids = gdf['h3_id'].tolist()

    # Extract numeric features (amenity counts, etc.)
    amenity_cols = extract_amenity_columns(gdf)
    if amenity_cols:
        X_counts = gdf[amenity_cols].fillna(0).to_numpy(dtype=float)
    else:
        X_counts = np.zeros((len(gdf), 1), dtype=float)

    # Optionally append centroid coordinates
    if include_coords:
        coords = compute_centroids(gdf)
        X = np.hstack([X_counts, coords])
    else:
        X = X_counts

    x = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor(build_h3_adjacency(h3_ids), dtype=torch.long)

    # Metadata and mappings
    props = [
        {k: row[k] for k in gdf.columns if k != 'geometry'}
        for _, row in gdf.iterrows()
    ]

    data = Data(x=x, edge_index=edge_index)
    data.h3_ids = h3_ids
    data.props = props
    data.feature_columns = amenity_cols
    data.resolution = int(gdf['resolution'].iloc[0]) if 'resolution' in gdf.columns else None

    print(f"✓ Loaded {len(h3_ids)} hexes")
    print(f"✓ Features: {len(amenity_cols)} amenity columns")
    print(f"✓ Edges: {edge_index.shape[1] // 2} (undirected)")
    return data


def save_graph(data: Data, out_path: str):
    """Save torch_geometric Data object with torch.save."""
    torch.save(data, out_path)
    print(f"✓ Saved graph to: {out_path}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert H3 hex GeoJSON into PyTorch Geometric graph for GraphMAE2 or similar models."
    )
    p.add_argument("--input", "-i", required=True, help="Input GeoJSON file (one resolution)")
    p.add_argument("--out", "-o", required=True, help="Output .pt file (e.g. graph_res8.pt)")
    p.add_argument("--no-coords", dest="coords", action="store_false",
                   help="Do not include centroid coordinates in node features")
    return p.parse_args()


def main_cli():
    args = parse_args()
    data = build_graph_from_geojson(args.input, include_coords=args.coords)
    save_graph(data, args.out)


if __name__ == "__main__":
    main_cli()
