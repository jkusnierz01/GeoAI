import pandas as pd
import h3
from srai.datasets import AirbnbMulticityDataset

RESOLUTION = 9
name = "Airbnb Multicity Dataset"

# --- AIRBNB CONFIGURATION ---
def prep_dataset(df):
    df['last_review'] = pd.to_datetime(df['last_review'])
    ref_date = pd.Timestamp('2024-01-01')
    df['days_since_review'] = (ref_date - df['last_review']).dt.days
    df['days_since_review'] = df['days_since_review'].fillna(3650)
    df['name_length'] = df['name'].str.len().fillna(0)

    df["h3_index"] = df.apply(lambda row: h3.latlng_to_cell(row.geometry.y, row.geometry.x, RESOLUTION), axis=1)

    return df

cats = ['room_type', 'city']
drop = ['id', 'host_id', 'name', 'host_name', 'neighbourhood', 'last_review', 'date']
dataset_loader = AirbnbMulticityDataset()
