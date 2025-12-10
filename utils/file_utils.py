import os
import pandas as pd
import numpy as np

def get_prefix(filename):
    """
    Extracts class/group prefix from filename.
    For files like 'abc_res7.geojson' → return 'abc'.
    """
    base = os.path.basename(filename)
    if "_res" in base:
        return base.split("_res")[0]
    return base.split(".")[0]

def load_and_merge_embeddings(df, path, prefix):
    """
    Loads embeddings from a file and merges them into the main dataframe.
    Handles both list-columns (e.g. [0.1, 0.2]) and wide-columns (0, 1, 2).
    """
    if not os.path.exists(path):
        print(f"WARNING: Embedding file not found at {path}. Skipping.")
        return df

    print(f"   -> Loading {prefix} embeddings from {path}...")
    
    # Load file (Try pickle first, then CSV)
    try:
        embed_df = pd.read_pickle(path)
    except:
        embed_df = pd.read_csv(path, index_col=0)

    # Ensure index is string to match h3_index
    embed_df.index = embed_df.index.astype(str)
    
    # CHECK FORMAT: Is it one column of lists? Or many columns of floats?
    # If it's a single column with lists/arrays inside
    if len(embed_df.columns) == 1 and isinstance(embed_df.iloc[0].iloc[0], (list, np.ndarray, np.generic)):
        col_name = embed_df.columns[0]
        # Expand list into separate columns
        # This can be slow for huge datasets
        expanded = pd.DataFrame(
            embed_df[col_name].tolist(), 
            index=embed_df.index
        )
        expanded = expanded.add_prefix(f"{prefix}_")
    else:
        # It's already wide format
        expanded = embed_df.add_prefix(f"{prefix}_")
    
    # Merge (Left join ensures we don't lose rows if embeddings are missing)
    # We join on the index of expanded (which should be h3) vs 'h3_index' column of df
    merged_df = df.join(expanded, on='h3_index', how='left')
    
    # Fill missing embeddings with 0 (for regions that didn't have an embedding)
    new_cols = [c for c in merged_df.columns if c.startswith(f"{prefix}_")]
    merged_df[new_cols] = merged_df[new_cols].fillna(0)
    
    print(f"      Added {len(new_cols)} features.")
    return merged_df
