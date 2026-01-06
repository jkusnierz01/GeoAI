import os
import csv
import rasterio
import h3
from rasterio.warp import transform
from tqdm import tqdm

def get_image_coordinates(tif_path):
    """
    Extracts the center latitude and longitude from a GeoTIFF file.
    """
    with rasterio.open(tif_path) as src:
        crs = src.crs
        
        bounds = src.bounds
        
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        
        lon, lat = transform(crs, 'EPSG:4326', [center_x], [center_y])
        
        return lat[0], lon[0]

def process_dataset_to_csv(dataset_root, output_csv, h3_resolution=8):
    """
    Walks through the dataset, calculates H3 index for each image, 
    and saves metadata to CSV.
    """
    
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset path '{dataset_root}' not found.")
        return

    data_rows = []
    
    valid_exts = ('.tif', '.tiff')

    print(f"Scanning dataset at: {dataset_root}")
    print(f"Using H3 Resolution: {h3_resolution}")

    for root, dirs, files in os.walk(dataset_root):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if file.lower().endswith(valid_exts):
                
                file_path = os.path.join(root, file)
                class_name = os.path.basename(root)
                
                try:
                    lat, lon = get_image_coordinates(file_path)
                    
                    h3_index = h3.latlng_to_cell(lat, lon, h3_resolution)
                    
                    data_rows.append({
                        'filename': file,
                        'class': class_name,
                        'latitude': lat,
                        'longitude': lon,
                        'h3_index': h3_index,
                        'h3_res': h3_resolution
                    })
                    
                except Exception as e:
                    print(f"Skipping {file}: {e}")

    if data_rows:
        keys = data_rows[0].keys()
        with open(output_csv, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data_rows)
        print(f"\nSuccess! Processed {len(data_rows)} images.")
        print(f"Results saved to: {os.path.abspath(output_csv)}")
    else:
        print("No valid images found.")

if __name__ == "__main__":
    DATASET_PATH = "EuroSAT_MS" 
    
    OUTPUT_FILE = "eurosat_h3_index.csv"
    
    RESOLUTION = 8

    process_dataset_to_csv(DATASET_PATH, OUTPUT_FILE, RESOLUTION)