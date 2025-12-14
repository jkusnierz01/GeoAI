import pandas as pd
import h3
from srai.datasets import PhiladelphiaCrimeDataset

name = "Philadelphia Crime"
dataset_loader = PhiladelphiaCrimeDataset()
dataset_loader.target = 'count'
RESOLUTION = 9

def prep_dataset(df):
    df["h3_index"] = df.apply(lambda row: h3.latlng_to_cell(row.geometry.y, row.geometry.x, RESOLUTION), axis=1)
    
    date_col = 'dispatch_date_time' if 'dispatch_date_time' in df.columns else 'dispatch_date'
    df['date'] = pd.to_datetime(df[date_col])
    df['hour'] = df['date'].dt.hour
    df['is_night'] = df['hour'].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)

    # 2. Aggregation
    agg_funcs = {
        'h3_index': 'count',    # Total Count
        'is_night': 'mean',     # Night Ratio
        'dc_dist': lambda x: x.mode()[0] if not x.mode().empty else "Unknown" # Dominant District
    }
    
    hex_data = df.groupby('h3_index').agg(agg_funcs).rename(columns={'h3_index': 'count'}).reset_index()

    type_col = 'text_general_code'
    
    top_crimes = ['Thefts', 'Vandalism/Criminal Mischief', 'Assaults', 'Narcotic / Drug Law Violations', 'Fraud']
    
    actual_top = df[type_col].value_counts().head(5).index.tolist()
    
    type_counts = df[df[type_col].isin(actual_top)].pivot_table(
        index='h3_index', 
        columns=type_col, 
        aggfunc='size', 
        fill_value=0
    ).add_prefix('type_')
    
    hex_data = hex_data.join(type_counts, on='h3_index').fillna(0)

    hex_data["lat"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
    hex_data["lon"] = hex_data["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])

    return hex_data

cats = ['dc_dist'] # The dominant district
drop = ['h3_index']