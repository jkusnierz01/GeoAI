import requests
import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import json
import time


# ==============================
# CONFIGURATION
# ==============================

OVERPASS_URL = "http://overpass-api.de/api/interpreter"
AMENITY_TYPES = ["school", "hospital", "restaurant"]  # ← add any others here
BOUNDING_BOXES = [
    (50.0, 14.0, 50.5, 14.5),
    (50.5, 14.5, 51.0, 15.0)
]
RESOLUTIONS = [7, 8, 9]


# ==============================
# FUNCTIONS
# ==============================

def fetch_overpass_data(amenity: str, bbox: tuple, retries: int = 3, delay: int = 10) -> dict:
    """Fetch data from Overpass API for a specific amenity type and bounding box."""
    query = f'''
    [out:json][timeout:60];
    node["amenity"="{amenity}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    out body;
    '''
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(OVERPASS_URL, params={'data': query}, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code in (504, 429):
                print(f"Attempt {attempt}/{retries} failed (HTTP {response.status_code}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
        except requests.exceptions.RequestException as e:
            print(f"Network error on attempt {attempt}/{retries}: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    raise ConnectionError("Overpass API request failed after multiple retries.")


def parse_overpass_points(data: dict, amenity: str) -> gpd.GeoDataFrame:
    """Convert Overpass data to a GeoDataFrame with amenity type."""
    points = [
        {'id': el['id'], 'amenity': amenity, 'geometry': Point(el['lon'], el['lat'])}
        for el in data.get('elements', [])
        if 'lat' in el and 'lon' in el
    ]
    return gpd.GeoDataFrame(points, geometry='geometry', crs='EPSG:4326') if points else gpd.GeoDataFrame(columns=['id', 'amenity', 'geometry'])


def h3_to_polygon(h3_id: str) -> Polygon:
    """Convert H3 hex ID to polygon."""
    boundary = h3.h3_to_geo_boundary(h3_id)
    coords = [(lon, lat) for lat, lon in boundary]
    return Polygon(coords)


def add_h3_indices(gdf: gpd.GeoDataFrame, resolutions: list[int]) -> gpd.GeoDataFrame:
    """Add H3 indices for each resolution."""
    for res in resolutions:
        gdf[f'h3_res_{res}'] = gdf.apply(
            lambda row: h3.geo_to_h3(row.geometry.y, row.geometry.x, res), axis=1
        )
    return gdf


def aggregate_amenities_by_h3(gdf: gpd.GeoDataFrame, resolution: int) -> gpd.GeoDataFrame:
    """Aggregate counts of each amenity per H3 hex."""
    grouped = (
        gdf.groupby([f'h3_res_{resolution}', 'amenity'])
        .size()
        .reset_index(name='count')
    )

    # Pivot: one row per h3_id, with amenity counts as columns
    pivot = grouped.pivot(index=f'h3_res_{resolution}', columns='amenity', values='count').fillna(0).reset_index()
    pivot['total_count'] = pivot[AMENITY_TYPES].sum(axis=1)

    # Add geometry
    pivot['geometry'] = pivot[f'h3_res_{resolution}'].apply(h3_to_polygon)
    pivot['resolution'] = resolution
    pivot['h3_id'] = pivot[f'h3_res_{resolution}']

    return gpd.GeoDataFrame(pivot, geometry='geometry', crs='EPSG:4326')


def gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """Convert GeoDataFrame to GeoJSON with cleaned coordinates."""
    geojson_data = json.loads(gdf.to_json())
    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates']
            if (len(coords) > 0 and isinstance(coords[0], list) and 
                len(coords[0]) > 0 and isinstance(coords[0][0], list) and
                len(coords[0][0]) > 0 and isinstance(coords[0][0][0], list)):
                feature['geometry']['coordinates'] = coords[0]
    return geojson_data


def save_geojson(gdf: gpd.GeoDataFrame, resolution: int, filename_prefix: str = "amenities_hexagons"):
    """Save GeoDataFrame to GeoJSON file."""
    geojson_data = gdf_to_geojson(gdf)
    filename = f"{filename_prefix}_res{resolution}.geojson"
    with open(filename, "w") as f:
        json.dump(geojson_data, f, indent=2)
    print(f"✓ Saved: {filename}")


# ==============================
# MAIN
# ==============================

def main():
    all_points = gpd.GeoDataFrame(columns=['id', 'amenity', 'geometry'])

    for amenity in AMENITY_TYPES:
        print(f"\nFetching data for amenity: {amenity}")
        for bbox in BOUNDING_BOXES:
            print(f"   → bbox: {bbox}")
            try:
                data = fetch_overpass_data(amenity, bbox)
                gdf = parse_overpass_points(data, amenity)
                all_points = pd.concat([all_points, gdf], ignore_index=True)
            except Exception as e:
                print(f"   Skipping bbox {bbox}: {e}")

    if all_points.empty:
        print("No data fetched at all.")
        return

    print("\nAdding H3 indices...")
    all_points = add_h3_indices(all_points, RESOLUTIONS)

    print("\nAggregating and saving results...")
    for res in RESOLUTIONS:
        agg = aggregate_amenities_by_h3(all_points, res)
        save_geojson(agg, res)

    print("\nDone!")


if __name__ == "__main__":
    main()
