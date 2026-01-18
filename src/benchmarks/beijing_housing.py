import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
import rootutils
import kagglehub
import os
import glob
import h3

# Setup root
ROOT = rootutils.setup_root(search_from=__file__, indicator=".project_root", pythonpath=True)

name = "beijing_housing"
RESOLUTION = 9

# Categorical columns
cats = [
    'buildingType', 
    'renovationCondition', 
    'buildingStructure', 
    'elevator', 
    'district', 
    'subway'
]

# Columns to drop
drop = [
    'url', 'id', 'Cid', 'tradeTime', 'DOM', 'floor'
]

class BeijingHousingLoader:
    def __init__(self, target="totalPrice"):
        self.target = target
        self.dataset_handle = "ruiqurm/lianjia"

    def load(self):
        print(f"Downloading/Loading dataset: {self.dataset_handle} via kagglehub...")
        
        # 1. Download
        dataset_path = kagglehub.dataset_download(self.dataset_handle)
        
        # 2. Find CSV
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
        target_file = None
        for f in csv_files:
            if "new.csv" in f:
                target_file = f
                break
        
        if not target_file and csv_files:
            target_file = csv_files[0]
        
        if not target_file:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")

        # 3. Load
        try:
            df = pd.read_csv(target_file, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(target_file, encoding='gb18030', low_memory=False)

        # 4. Clean Coordinates & Create Geometry
        df = df.dropna(subset=['Lng', 'Lat'])
        # Sanity check for Beijing coordinates
        df = df[(df['Lng'] > 70) & (df['Lng'] < 140) & (df['Lat'] > 10) & (df['Lat'] < 60)]
        
        geometry = [Point(xy) for xy in zip(df.Lng, df.Lat)]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        # 5. Basic Cleaning
        gdf['constructionTime'] = pd.to_numeric(gdf['constructionTime'], errors='coerce')
        gdf['constructionTime'] = gdf['constructionTime'].fillna(gdf['constructionTime'].median())

        # Drop rows where target is NaN
        gdf = gdf.dropna(subset=[self.target])
        
        # 6. Leakage Removal
        if self.target == 'totalPrice' and 'price' in gdf.columns:
            gdf = gdf.drop(columns=['price'])
        elif self.target == 'price' and 'totalPrice' in gdf.columns:
            gdf = gdf.drop(columns=['totalPrice'])

        # 7. Split
        train_gdf, test_gdf = train_test_split(gdf, test_size=0.2, random_state=42)

        return {
            "train": train_gdf,
            "test": test_gdf
        }

def prep_dataset(gdf):
    """
    Benchmark-specific preprocessing applied in run_experiment.
    """
    gdf = gdf.copy()
    
    # 1. Handle Numerics
    numeric_cols = ['square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'communityAverage']
    for col in numeric_cols:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(0)

    # 2. Feature Engineering (Building Age)
    current_year = 2024
    if 'constructionTime' in gdf.columns:
        gdf['building_age'] = current_year - gdf['constructionTime']
        # We drop constructionTime here so it doesn't get encoded or scaled improperly later
        gdf = gdf.drop(columns=['constructionTime'])

    # 3. H3 Indexing (Requested)
    # Using list comprehension for speed over .apply
    gdf["h3_index"] = [
        h3.latlng_to_cell(y, x, RESOLUTION) 
        for x, y in zip(gdf.geometry.x, gdf.geometry.y)
    ]

    return gdf

dataset_loader = BeijingHousingLoader()