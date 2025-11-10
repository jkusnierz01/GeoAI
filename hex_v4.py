import requests
import h3
import geopandas as gpd
from shapely.geometry import Point, Polygon
import json
import pandas as pd
import time

QUERIES = {
    'health': [
        ('amenity', 'hospital'),
        ('amenity', 'pharmacy'),
    ],
    'services': [
        ('amenity', 'bank'),
        ('amenity', 'police'),
    ],
    'shops': [
        ('shop', 'supermarket'),
        ('shop', 'bakery'),
        ('shop', 'greengrocer'),
        ('shop', 'alcohol'),
        ('shop', 'clothes'),
    ],
    'food_drink': [
        ('amenity', 'restaurant'),
        ('amenity', 'bar'),
        ('amenity', 'nightclub'),
    ],
    'tourism': [
        ('tourism', 'hotel'),
        ('tourism', 'museum'),
    ],
    'landuse': [
        ('landuse', 'cemetery'),
        ('landuse', 'industrial'),
    ],
    'leisure': [
        ('leisure', 'park'),
        ('leisure', 'sports_centre'),
        ('leisure', 'playground'),
    ],
    'buildings': [
        ('building', 'office'),
        ('building', 'house'),
        ('building', 'apartments'),
    ],
    'transport': [
        ('railway', 'station'),
        ('railway', 'tram_stop'),
        ('highway', 'bus_stop'),
        ('aeroway', 'aerodrome'),
    ],
    'culture': [
        ('amenity', 'cinema'),
        ('amenity', 'theatre'),
        ('amenity', 'library'),
        ('amenity', 'place_of_worship'),
        ('amenity', 'school'),
    ],
}

def fetch_category(category_name, tags, bbox, retry=0, max_retries=5):
    """
    Pobierz dane dla jednej kategorii z pełnym debugowaniem
    """
    if retry > max_retries:
        print(f"✗ Przekroczono limit prób")
        return []
    
    min_lat, min_lon, max_lat, max_lon = bbox
    overpass_url = "http://overpass-api.de/api/interpreter"
    
   
    if min_lat >= max_lat or min_lon >= max_lon:
        print(f"✗ Błędny bbox: ({min_lat}, {min_lon}, {max_lat}, {max_lon})")
        return []
    
    
    query_parts = []
    for key, value in tags:
        query_parts.append(f'nwr["{key}"="{value}"]({min_lat}, {min_lon}, {max_lat}, {max_lon});')
    
    overpass_query = f'''
[out:json][timeout:500];
(
  {chr(10).join(query_parts)}
);
out center;
'''
    
    print(f"⏳ {category_name}...", end=" ")
    
    try:
        response = requests.get(
            overpass_url,
            params={'data': overpass_query},
            timeout=120,
            headers={'User-Agent': 'GEO_AI/1.0'}
        )
        
        
        if response.status_code == 429:
            wait_time = 30 + (10 * retry)  
            print(f"⏱️  Rate limit - czekam {wait_time}s...")
            time.sleep(wait_time)
            return fetch_category(category_name, tags, bbox, retry + 1, max_retries)
        
        
        if response.status_code == 504:
            wait_time = 20 + (10 * retry)  
            print(f"⏱️  504 Gateway Timeout - czekam {wait_time}s...")
            time.sleep(wait_time)
            return fetch_category(category_name, tags, bbox, retry + 1, max_retries)
        
        
        if response.status_code == 503:
            wait_time = 60 + (20 * retry) 
            print(f"⏱️  503 Service Unavailable - czekam {wait_time}s...")
            time.sleep(wait_time)
            return fetch_category(category_name, tags, bbox, retry + 1, max_retries)
        
        if response.status_code != 200:
            print(f"✗ HTTP {response.status_code}")
            return []
        
        data = response.json()
        count = len(data.get('elements', []))
        print(f"✓ {count} elementów")
        
        time.sleep(2)
        return data.get('elements', [])
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON Error: {e}")
        print(f"   Response text: {response.text[:200] if response else 'None'}")
        return []
    except requests.exceptions.Timeout:
        print(f"✗ Timeout")
        return []
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error")
        return []
    except Exception as e:
        print(f"✗ {type(e).__name__}: {e}")
        return []


bbox = (24.51, 46.42, 24.96, 47.01)
#       min_lat  min_lon  max_lat  max_lon

all_elements = []

print(f"\n📍 Obszar: Lat({bbox[0]:.2f}, {bbox[2]:.2f}) Lon({bbox[1]:.2f}, {bbox[3]:.2f})\n")

for category_name, tags in QUERIES.items():
    elements = fetch_category(category_name, tags, bbox)
    all_elements.extend(elements)
    print() 

print(f"\n{'='*50}")
print(f"✓ Razem pobrano: {len(all_elements)} elementów")
print(f"{'='*50}\n")


TAG_MAPPING = {
    'amenity': {
        'school', 'hospital', 'pharmacy', 'bank', 'restaurant', 
        'bar', 'nightclub', 'police', 'cinema', 'theatre', 
        'library', 'place_of_worship'
    },
    'shop': {
        'supermarket', 'bakery', 'greengrocer', 'alcohol', 'clothes'
    },
    'tourism': {
        'hotel', 'museum'
    },
    'landuse': {
        'cemetery', 'industrial'
    },
    'leisure': {
        'park', 'sports_centre', 'playground'
    },
    'building': {
        'office', 'house', 'apartments'
    },
    'railway': {
        'station', 'tram_stop'
    },
    'highway': {
        'bus_stop'
    },
    'aeroway': {
        'aerodrome'
    }
}

processed_data = []

for element in all_elements:
    tags = element.get('tags', {})
    
    if not tags:
        continue

    lat, lon = None, None
    main_category, specific_type = None, None

    for key, value in tags.items():
        if key in TAG_MAPPING and value in TAG_MAPPING[key]:
            main_category = key
            specific_type = value
            break  

    if not main_category:
        continue

    if element['type'] == 'node':
        lat = element.get('lat')
        lon = element.get('lon')
    elif 'center' in element:
        lat = element['center'].get('lat')
        lon = element['center'].get('lon')

    if lat is not None and lon is not None:
        processed_data.append({
            'geometry': Point(lon, lat),
            'id': element['id'],
            'object_type': f'{main_category}_{specific_type}'
        })

gdf = gpd.GeoDataFrame(processed_data, crs="EPSG:4326")

print(f"✓ Przetworzono: {len(gdf)} obiektów")
print(f"✓ Typy: {gdf['object_type'].nunique()}\n")
print(gdf.head())

resolutions = [7, 8, 9]

for resolution in resolutions:
    gdf[f'h3_res_{resolution}'] = gdf.apply(
        lambda row: h3.geo_to_h3(row.geometry.y, row.geometry.x, resolution), 
        axis=1
    )

def h3_to_polygon(h3_id):
    boundary = h3.h3_to_geo_boundary(h3_id)
    coords = [(lon, lat) for lat, lon in boundary]
    return Polygon(coords)

for resolution in resolutions:
    counts_by_type = gdf.groupby([f'h3_res_{resolution}', 'object_type']).size()
    counts_wide = counts_by_type.unstack(fill_value=0)
    counts_wide = counts_wide.reset_index()
    counts_wide['geometry'] = counts_wide[f'h3_res_{resolution}'].apply(h3_to_polygon)
    counts_wide = gpd.GeoDataFrame(counts_wide, geometry='geometry', crs='EPSG:4326')
    counts_wide['resolution'] = resolution
    counts_wide = counts_wide.rename(columns={f'h3_res_{resolution}': 'h3_id'})

    geojson_data = json.loads(counts_wide.to_json(drop_id=True))

    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates']
            if len(coords) > 0 and isinstance(coords[0][0][0], list):
                feature['geometry']['coordinates'] = coords[0]

    filename = f'riyadh_hexagons_res{resolution}.geojson'
    with open(filename, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    print(f"✓ Zapisano: {filename}")