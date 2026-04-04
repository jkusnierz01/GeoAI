import argparse
import os
import requests
import h3
import pandas as pd
import math
import time
import re
from tqdm import tqdm

def generate_grid_points(center_lat, center_lon, radius_m, step_m=4000):
    """Generates a grid of coordinates covering a bounding box around the center."""
    points = []
    lat_degree_m = 111320
    lon_degree_m = 111320 * math.cos(math.radians(center_lat))
    
    lat_offset = radius_m / lat_degree_m
    lon_offset = radius_m / lon_degree_m
    
    lat_start = center_lat - lat_offset
    lat_end = center_lat + lat_offset
    lon_start = center_lon - lon_offset
    lon_end = center_lon + lon_offset
    
    lat_step = step_m / lat_degree_m
    lon_step = step_m / lon_degree_m
    
    lat = lat_start
    while lat <= lat_end + (lat_step / 2):
        lon = lon_start
        while lon <= lon_end + (lon_step / 2):
            dist = math.sqrt(((lat - center_lat) * lat_degree_m)**2 + ((lon - center_lon) * lon_degree_m)**2)
            if dist <= radius_m + step_m:
                points.append((lat, lon))
            lon += lon_step
        lat += lat_step
        
    return points

def fetch_wikivoyage_point(session, lat, lon, search_radius=5000, limit=500):
    """Fetches Wikivoyage district/city guides for a specific coordinate."""
    # Changed endpoint to Wikivoyage
    geo_url = "https://en.wikivoyage.org/w/api.php"
    
    geo_params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": search_radius, 
        "gslimit": limit,
        "format": "json"
    }
    
    try:
        response = session.get(url=geo_url, params=geo_params)
        if response.status_code != 200:
            return []
            
        articles = response.json().get('query', {}).get('geosearch', [])
        if not articles:
            return []

        results = []
        for article in articles:
            page_id = article['pageid']
            text_params = {
                "action": "query",
                "prop": "extracts",
                "exintro": False, # We want the full text so we can parse sections
                "explaintext": True, # Pure raw text, no HTML
                "pageids": page_id,
                "format": "json"
            }
            
            text_res = session.get(url=geo_url, params=text_params).json()
            extract = text_res.get('query', {}).get('pages', {}).get(str(page_id), {}).get('extract', '')
            
            if extract and len(extract.strip()) > 50: 
                # WIKIVOYAGE VIBE EXTRACTOR:
                # We split the text at the first instance of '== Get in ==', '== See ==', or '== Get around =='
                # This perfectly isolates the Intro and "Understand" sections which describe the area's look and feel.
                vibe_text = re.split(r'==\s*(Get in|See|Do|Get around)\s*==', extract, flags=re.IGNORECASE)[0]
                
                # Clean up newlines for the CSV
                vibe_text = vibe_text.strip().replace('\n', ' ')
                
                if len(vibe_text) > 20:
                    results.append({
                        'pageid': page_id,
                        'title': article['title'],
                        'latitude': article['lat'],
                        'longitude': article['lon'],
                        'text': vibe_text
                    })
        return results
    except Exception as e:
        print(f"  [!] Failed at {lat:.4f}, {lon:.4f}: {e}")
        return []

def fetch_batch(lat, lon, total_radius):
    """Manages the grid search to bypass the radius and item limits."""
    grid_points = generate_grid_points(lat, lon, total_radius, step_m=4000)
    print(f"Generated {len(grid_points)} grid points to scan for {total_radius}m radius.")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'LeonGeoAIResearch/1.0 (263495@student.pwr.edu.pl) python-requests'
    })
    
    all_results = []
    
    for point_lat, point_lon in tqdm(grid_points, desc="Scanning Grid (Wikivoyage)"):
        results = fetch_wikivoyage_point(session, point_lat, point_lon, search_radius=5000)
        all_results.extend(results)
        time.sleep(0.1) 
        
    df_results = pd.DataFrame(all_results)
    
    if not df_results.empty:
        original_count = len(df_results)
        df_results = df_results.drop_duplicates(subset=['pageid'])
        print(f"Scraped {original_count} guides. Unique areas found: {len(df_results)}.")
        
    return df_results

def process_to_h3(df, min_res, max_res):
    """Maps coordinates to H3 and aggregates text."""
    if df.empty:
        return pd.DataFrame()

    print(f"Mapping coordinates to H3 Resolutions {min_res} through {max_res}...")
    res_columns = []
    for res in range(min_res, max_res + 1):
        col_name = f'h3res{res}'
        # Adjust latlng_to_cell/geo_to_h3 based on your h3-py version
        df[col_name] = df.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], res), 
            axis=1
        )
        res_columns.append(col_name)

    print(f"Aggregating raw text by finest resolution ({max_res})...")
    aggregated_df = df.groupby(res_columns).agg(
        text_content=('text', lambda texts: " [SEP] ".join(texts)),
        area_count=('title', 'count'),
        area_titles=('title', lambda titles: ", ".join(titles))
    ).reset_index()

    return aggregated_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch download Urban descriptions from Wikivoyage.")
    parser.add_argument("--input_file", type=str, default="cities.csv", help="CSV with columns: city, lat, lon, radius")
    parser.add_argument("--out", type=str, default="data/text", help="Output folder")
    parser.add_argument("--min_res", type=int, default=7, help="Minimum H3 resolution")
    parser.add_argument("--max_res", type=int, default=9, help="Maximum H3 resolution")
    
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.input_file):
        print(f"Error: Could not find '{args.input_file}'.")
        exit(1)

    df_cities = pd.read_csv(args.input_file)
    
    for index, row in df_cities.iterrows():
        city = str(row['city'])
        lat = float(row['lat'])
        lon = float(row['lon'])
        rad = int(row['radius'])

        print("\n" + "="*60)
        print(f"Processing Vibe: {city.upper()} (Lat: {lat}, Lon: {lon}, Target Radius: {rad}m)")
        print("="*60)
        
        try:
            df_wiki = fetch_batch(lat, lon, rad)
            
            if not df_wiki.empty:
                df_h3_text = process_to_h3(df_wiki, args.min_res, args.max_res)
                
                safe_city_name = city.lower().replace(" ", "_")
                filename = os.path.join(args.out, f"{safe_city_name}_wikivoyage_h3_res{args.min_res}_to_{args.max_res}.csv")
                
                df_h3_text.to_csv(filename, index=False, encoding='utf-8')
                print(f"Success! Processed {len(df_h3_text)} unique H3 blocks. Saved to: {filename}")
            else:
                print(f"Skipped {city}: No guides found.")
                
        except Exception as e:
            print(f"An error occurred while processing {city}: {e}")
            continue
            
    print("\nBatch processing complete!")