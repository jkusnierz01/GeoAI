import argparse
import os
import requests
import h3
import pandas as pd
import math
import time
from tqdm import tqdm

def generate_grid_points(center_lat, center_lon, radius_m, step_m=4000):
    """Generates a grid of coordinates covering a bounding box around the center."""
    points = []
    # Approximate conversions: 1 degree latitude is ~111.32 km
    lat_degree_m = 111320
    lon_degree_m = 111320 * math.cos(math.radians(center_lat))
    
    # Bounding box offsets
    lat_offset = radius_m / lat_degree_m
    lon_offset = radius_m / lon_degree_m
    
    lat_start = center_lat - lat_offset
    lat_end = center_lat + lat_offset
    lon_start = center_lon - lon_offset
    lon_end = center_lon + lon_offset
    
    # Step sizes in degrees
    lat_step = step_m / lat_degree_m
    lon_step = step_m / lon_degree_m
    
    lat = lat_start
    while lat <= lat_end + (lat_step / 2):
        lon = lon_start
        while lon <= lon_end + (lon_step / 2):
            # Only add points that are roughly within our overall target radius
            # We add step_m to the radius to ensure we cover the edges fully
            dist = math.sqrt(((lat - center_lat) * lat_degree_m)**2 + ((lon - center_lon) * lon_degree_m)**2)
            if dist <= radius_m + step_m:
                points.append((lat, lon))
            lon += lon_step
        lat += lat_step
        
    return points

def fetch_single_point(session, lat, lon, search_radius=5000, limit=500):
    """Fetches articles for a single specific coordinate point."""
    geo_url = "https://en.wikipedia.org/w/api.php"
    
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
        # Fetch summaries for found articles
        for article in articles:
            page_id = article['pageid']
            text_params = {
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "pageids": page_id,
                "format": "json"
            }
            
            text_res = session.get(url=geo_url, params=text_params).json()
            extract = text_res.get('query', {}).get('pages', {}).get(str(page_id), {}).get('extract', '')
            
            if extract and len(extract.strip()) > 20: 
                results.append({
                    'pageid': page_id,
                    'title': article['title'],
                    'latitude': article['lat'],
                    'longitude': article['lon'],
                    'text': extract.strip()
                })
        return results
    except Exception as e:
        print(f"  [!] Failed at {lat:.4f}, {lon:.4f}: {e}")
        return []

def fetch_wikipedia_batch(lat, lon, total_radius):
    """Manages the grid search to bypass Wikipedia's radius and item limits."""
    # We step every 4km, and search a 5km overlapping circle. 
    # This guarantees no gaps and stays well below the 10km API limit.
    grid_points = generate_grid_points(lat, lon, total_radius, step_m=4000)
    print(f"Generated {len(grid_points)} grid points to scan for {total_radius}m radius.")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'LeonGeoAIResearch/1.0 (263495@student.pwr.edu.pl) python-requests'
    })
    
    all_results = []
    
    # Tqdm progress bar for our grid points
    for point_lat, point_lon in tqdm(grid_points, desc="Scanning Grid"):
        results = fetch_single_point(session, point_lat, point_lon, search_radius=5000)
        all_results.extend(results)
        
        # Be polite to Wikipedia's servers to prevent getting IP banned
        time.sleep(0.1) 
        
    df_results = pd.DataFrame(all_results)
    
    if not df_results.empty:
        # Crucial step: Drop duplicates because our circles overlap!
        original_count = len(df_results)
        df_results = df_results.drop_duplicates(subset=['pageid'])
        print(f"Scraped {original_count} articles. After removing overlaps: {len(df_results)} unique articles found.")
        
    return df_results

def process_to_h3(df, min_res, max_res):
    """Maps coordinates to multiple H3 resolutions and aggregates text."""
    if df.empty:
        return pd.DataFrame()

    print(f"Mapping coordinates to H3 Resolutions {min_res} through {max_res}...")
    res_columns = []
    for res in range(min_res, max_res + 1):
        col_name = f'h3res{res}'
        df[col_name] = df.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], res), 
            axis=1
        )
        res_columns.append(col_name)

    print(f"Aggregating text by finest resolution ({max_res})...")
    aggregated_df = df.groupby(res_columns).agg(
        text_content=('text', lambda texts: " [SEP] ".join(texts)),
        article_count=('title', 'count'),
        article_titles=('title', lambda titles: ", ".join(titles))
    ).reset_index()

    return aggregated_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch download Wikipedia articles using grid scanning.")
    
    parser.add_argument("--input_file", type=str, default="cities.csv", help="CSV with columns: city, lat, lon, radius")
    parser.add_argument("--out", type=str, default="data/text", help="Output folder")
    parser.add_argument("--min_res", type=int, default=7, help="Minimum H3 resolution (default: 7)")
    parser.add_argument("--max_res", type=int, default=9, help="Maximum H3 resolution (default: 9)")
    
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.input_file):
        print(f"Error: Could not find input file '{args.input_file}'.")
        exit(1)

    df_cities = pd.read_csv(args.input_file)
    
    for index, row in df_cities.iterrows():
        city = str(row['city'])
        lat = float(row['lat'])
        lon = float(row['lon'])
        rad = int(row['radius'])

        print("\n" + "="*60)
        print(f"Processing: {city.upper()} (Lat: {lat}, Lon: {lon}, Target Radius: {rad}m)")
        print("="*60)
        
        try:
            # 1. Scrape Wikipedia using our new grid search
            df_wiki = fetch_wikipedia_batch(lat, lon, rad)
            
            if not df_wiki.empty:
                # 2. Map to H3 and Aggregate
                df_h3_text = process_to_h3(df_wiki, args.min_res, args.max_res)
                
                # 3. Save to CSV
                safe_city_name = city.lower().replace(" ", "_")
                filename = os.path.join(args.out, f"{safe_city_name}_wiki_h3_res{args.min_res}_to_{args.max_res}.csv")
                
                df_h3_text.to_csv(filename, index=False, encoding='utf-8')
                
                print(f"Success! Processed {len(df_h3_text)} unique H3 blocks at resolution {args.max_res}.")
                print(f"Data saved locally to: {filename}")
            else:
                print(f"Skipped {city}: No articles found.")
                
        except Exception as e:
            print(f"An error occurred while processing {city}: {e}")
            continue
            
    print("\nBatch processing complete!")