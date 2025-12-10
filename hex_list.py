import requests
import h3
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import json
import time


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


ALL_OBJECT_TYPES = sorted([
    f"{key}_{value}" 
    for key, values in TAG_MAPPING.items() 
    for value in values
])


def fetch_data_from_h3_list(h3_ids, resolution, output_filename=None):
    
    if not h3_ids:
        print("Lista pusta")
        return None
    
    
    all_lats = []
    all_lons = []
    
    for h3_id in h3_ids:
        try:
            boundary = h3.h3_to_geo_boundary(h3_id)
            for lat, lon in boundary:
                all_lats.append(lat)
                all_lons.append(lon)
        except Exception as e:
            print(f"Błąd: {e}")
            continue
    
    if not all_lats or not all_lons:
        return None
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    print(f"Bbox: ({min_lat:.4f}, {min_lon:.4f}) → ({max_lat:.4f}, {max_lon:.4f})\n")
    
    query_parts = []
    for key, values in TAG_MAPPING.items():
        for value in values:
            query_parts.append(f'nwr["{key}"="{value}"]({min_lat},{min_lon},{max_lat},{max_lon});')
    
    overpass_query = f'''
[out:json][timeout:600];
(
  {chr(10).join(query_parts)}
);
out center;
'''
    
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    max_retries = 5
    
    for retry in range(max_retries):
        try:
            response = requests.get(
                overpass_url,
                params={'data': overpass_query},
                timeout=300,
                headers={'User-Agent': 'GEO_AI/1.0'}
            )
            
            if response.status_code == 200:
                break
            elif response.status_code in [429, 503, 504]:
                wait_time = 30 + (20 * retry)
                print(f"HTTP {response.status_code} - czekam {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"HTTP {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Timeout - próba {retry + 1}/{max_retries}")
            time.sleep(30)
        except Exception as e:
            print(f"Błąd: {e}")
            return None
    else:
        print("Przekroczono limit prób")
        return None
    
    data = response.json()
    elements = data.get('elements', [])
    print(f"Pobrano: {len(elements)} elementów\n")
    
    processed_data = []
    
    for element in elements:
        tags = element.get('tags', {})
        if not tags:
            continue
        
        object_type = None
        for key, value in tags.items():
            if key in TAG_MAPPING and value in TAG_MAPPING[key]:
                object_type = f'{key}_{value}'
                break
        
        if not object_type:
            continue
        
        if element['type'] == 'node':
            lat, lon = element.get('lat'), element.get('lon')
        elif 'center' in element:
            lat, lon = element['center'].get('lat'), element['center'].get('lon')
        else:
            continue
        
        if lat and lon:
            processed_data.append({
                'lat': lat,
                'lon': lon,
                'object_type': object_type
            })
    
    h3_set = set(h3_ids)
    hex_counts = {h3_id: {ot: 0 for ot in ALL_OBJECT_TYPES} for h3_id in h3_ids}
    
    for item in processed_data:
        h3_id = h3.geo_to_h3(item['lat'], item['lon'], resolution)
        if h3_id in h3_set:
            hex_counts[h3_id][item['object_type']] += 1
    
    print(f"Przetworzono: {len(processed_data)} obiektów")
    
    
    features = []
    
    for h3_id in h3_ids:
        
        boundary = h3.h3_to_geo_boundary(h3_id)
    
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])
        
        properties = {
            "h3_id": h3_id
        }
        
        for obj_type in ALL_OBJECT_TYPES:
            properties[obj_type] = hex_counts[h3_id][obj_type]
        

        properties["resolution"] = resolution
        
    
        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            }
        }
        
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    if output_filename is None:
        output_filename = f'h3_hexagons_res{resolution}.geojson'
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    
    print(f"Zapisano: {output_filename}")
    print(f"Hexów: {len(features)}")
  
    
    gdf = gpd.read_file(output_filename)
    return gdf


if __name__ == "__main__":
    
    h3_ids_example = [
        '892e2980103ffff',
        '892e2980107ffff', 
        '892e2980123ffff',
    ]

    detected_resolution = h3.h3_get_resolution(h3_ids_example[0])

    result = fetch_data_from_h3_list(
        h3_ids=h3_ids_example,
        resolution=detected_resolution,
        output_filename='test_hexagons.geojson'
    )
    
    if result is not None:
        print(result.head())