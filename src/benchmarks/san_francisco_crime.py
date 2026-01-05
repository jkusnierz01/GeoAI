import pandas as pd
import h3
from srai.datasets import PoliceDepartmentIncidentsDataset

name = "SF Crime"
dataset_loader = PoliceDepartmentIncidentsDataset()
dataset_loader.target = 'count'
RESOLUTION = 10

def prep_dataset(df):
    df["h3_index"] = df.apply(lambda row: h3.latlng_to_cell(row.geometry.y, row.geometry.x, RESOLUTION), axis=1)
    
    df['incident_datetime'] = pd.to_datetime(df['Incident Datetime'], format='mixed')
    
    df['hour'] = df['incident_datetime'].dt.hour
    df['is_night'] = df['hour'].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)
    df['is_weekend'] = df['incident_datetime'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    agg_funcs = {
        'h3_index': 'count',           
        'is_night': 'mean',            
        'is_weekend': 'mean',          
        'Police District': lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    }
    
    hex_data = df.groupby('h3_index').agg(agg_funcs).rename(columns={'h3_index': 'count'}).reset_index()

    top_crimes = ['Larceny Theft', 'Malicious Mischief', 'Assault', 'Motor Vehicle Theft', 'Non-Criminal']
    
    type_counts = df[df['Incident Category'].isin(top_crimes)].pivot_table(
        index='h3_index', 
        columns='Incident Category', 
        aggfunc='size', 
        fill_value=0
    ).add_prefix('type_')
    
    hex_data = hex_data.join(type_counts, on='h3_index').fillna(0)

    hex_data["lat"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
    hex_data["lon"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])

    return hex_data

cats = ['Police District']
drop = ['h3_index']