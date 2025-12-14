import os

def get_prefix(filename):
    """
    Extracts class/group prefix from filename.
    For files like 'abc_res7.geojson' → return 'abc'.
    """
    base = os.path.basename(filename)
    if "_res" in base:
        return base.split("_res")[0]
    return base.split(".")[0]

import pandas as pd
import pickle

def load_and_merge_embeddings(df, embedding_path, prefix="emb"):
    """
    Loads embeddings from a pickle file (dict: h3_id -> embedding)
    and merges them into the dataframe on 'h3_index'.
    """
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
        
    # Convert to DataFrame
    emb_df = pd.DataFrame.from_dict(embeddings, orient='index')
    emb_df.columns = [f"{prefix}_{i}" for i in range(emb_df.shape[1])]
    emb_df.index.name = 'h3_index'

    # Merge
    merged_df = df.merge(emb_df, on='h3_index', how='left')
    
    # Check match rate
    matched_count = merged_df[f"{prefix}_0"].notna().sum()
    total_count = len(merged_df)
    print(f"Merged {prefix} embeddings: {matched_count}/{total_count} rows matched ({matched_count/total_count:.1%})")

    # Fill NaNs with 0 (for nodes without embeddings)
    cols = [c for c in merged_df.columns if c.startswith(prefix)]
    merged_df[cols] = merged_df[cols].fillna(0)
    
    return merged_df
