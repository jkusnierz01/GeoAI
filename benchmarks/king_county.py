import pandas as pd
from srai.datasets import HouseSalesInKingCountyDataset

name = "King County House Sales Dataset"

# --- KING COUNTY CONFIGURATION ---
def prep_dataset(df):
    df['date'] = pd.to_datetime(df['date'])
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month
    
    current_year = 2016
    df['house_age'] = current_year - df['yr_built']
    
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
    
    return df

cats = ['zipcode'] 
drop = ['id', 'date']
dataset_loader = HouseSalesInKingCountyDataset()