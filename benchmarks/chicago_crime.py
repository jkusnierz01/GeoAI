import pandas as pd
import h3
from srai.datasets import ChicagoCrimeDataset

name = "Chicago Crime Dataset"
dataset_loader = ChicagoCrimeDataset()
dataset_loader.target = 'count' 
RESOLUTION = 9

def prep_dataset(df):
    df["h3_index"] = df.apply(lambda row: h3.latlng_to_cell(row.geometry.y, row.geometry.x, RESOLUTION), axis=1)
    
    date_col = 'Date' if 'Date' in df.columns else 'date'
    
    df['date_clean'] = pd.to_datetime(df[date_col], format='mixed')
    df['hour'] = df['date_clean'].dt.hour
    
    df['is_night'] = df['hour'].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)

    arrest_col = 'Arrest' if 'Arrest' in df.columns else 'arrest'
    if arrest_col in df.columns:
        df['arrest_flag'] = df[arrest_col].astype(int)
    else:
        df['arrest_flag'] = 0

    dist_col = 'District' if 'District' in df.columns else 'district'

    agg_funcs = {
        'h3_index': 'count',           
        'is_night': 'mean',            
        'arrest_flag': 'mean',         
        dist_col: lambda x: x.mode()[0] if not x.mode().empty else "Unknown" 
    }
    
    hex_data = df.groupby('h3_index').agg(agg_funcs).rename(columns={'h3_index': 'count'}).reset_index()

    type_col = 'Primary Type' if 'Primary Type' in df.columns else 'primary_type'
    
    top_crimes = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT']
    
    type_counts = df[df[type_col].isin(top_crimes)].pivot_table(
        index='h3_index', 
        columns=type_col, 
        aggfunc='size', 
        fill_value=0
    ).add_prefix('type_')
    
    hex_data = hex_data.join(type_counts, on='h3_index').fillna(0)

    hex_data["lat"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
    hex_data["lon"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])

    if dist_col != 'district':
        hex_data.rename(columns={dist_col: 'district'}, inplace=True)

    return hex_data

cats = ['district']
drop = ['h3_index']
