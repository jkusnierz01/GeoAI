import os
import torch
import glob
import re

# input_dir = "src/benchmarks/graphs_old"
# output_dir = "src/benchmarks/graphs"


input_dir = "dataset"
output_dir = "dataset_aligned"

os.makedirs(output_dir, exist_ok=True)

pt_files = glob.glob(os.path.join(input_dir, "*.pt"))

print(f"Found {len(pt_files)} files.")


# --- Determine expected feature dimension (majority) and feature columns ---
feature_dims = []
feature_columns_list = []
for file_path in pt_files:
    data = torch.load(file_path, weights_only=False)
    feature_dims.append(data.x.shape[1])
    feature_columns_list.append(getattr(data, 'feature_columns', None))
from collections import Counter
dim_counter = Counter(feature_dims)
expected_dim = dim_counter.most_common(1)[0][0]
print(f"[clean_graphs.py] Expected feature dimension (majority): {expected_dim}")

# Find the most common feature_columns set (as reference)
columns_counter = Counter([tuple(cols) if cols is not None else None for cols in feature_columns_list])
expected_columns = None
if columns_counter:
    expected_columns = columns_counter.most_common(1)[0][0]
    if expected_columns is not None:
        expected_columns = list(expected_columns)
print(f"[clean_graphs.py] Using feature columns reference: {expected_columns}")

def get_res(filename):
    match = re.search(r'res(\d+)', filename)
    return int(match.group(1)) if match else 8

# --- PASS 1: Calculate Global Stats (Excluding is_empty) ---
global_min = None
global_max = None

print("Pass 1: Calculating Global Stats for Amenities...")
for file_path in pt_files:
    data = torch.load(file_path, weights_only=False)
    x = data.x.float()
    
    is_empty_idx = int(data.is_empty_index) if hasattr(data, 'is_empty_index') else None
    
    # Select only columns that are NOT is_empty
    feat_mask = [i for i in range(x.shape[1]) if i != is_empty_idx]
    x_amenities = x[:, feat_mask]
    
    # Log transform counts to squash outliers
    x_amenities = torch.log1p(torch.clamp(x_amenities, min=0.0))

    if global_min is None:
        global_min = x_amenities.min(dim=0).values
        global_max = x_amenities.max(dim=0).values
    else:
        global_min = torch.minimum(global_min, x_amenities.min(dim=0).values)
        global_max = torch.maximum(global_max, x_amenities.max(dim=0).values)

# --- PASS 2: Transform Amenities, Keep is_empty Binary ---
print("Pass 2: Normalizing and adding Resolution feature...")
eps = 1e-8


for file_path in pt_files:
    data = torch.load(file_path, weights_only=False)
    file_name = os.path.basename(file_path)
    x = data.x.float()

    # 1. Identify all *_index attributes (feature names and their indices)
    feature_indices = {}
    for attr in dir(data):
        if attr.endswith('_index') and not attr.startswith('__'):
            try:
                idx = int(getattr(data, attr))
                feature_indices[attr.replace('_index', '')] = idx
            except Exception:
                pass

    # 2. Coordinate Removal (if not already done)
    for feat in ['centroid_x', 'centroid_y']:
        idx_attr = f'{feat}_index'
        if hasattr(data, idx_attr):
            idx = int(getattr(data, idx_attr))
            x = torch.cat([x[:, :idx], x[:, idx+1:]], dim=1)
            delattr(data, idx_attr)


    is_empty_idx = int(data.is_empty_index) if hasattr(data, 'is_empty_index') else None

    # If is_empty is missing, add a column of zeros and set is_empty_index
    if is_empty_idx is None or is_empty_idx >= x.shape[1]:
        # Add is_empty as last column (all zeros)
        is_empty_col = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        x = torch.cat([x, is_empty_col], dim=1)
        new_idx = x.shape[1] - 1
        data.is_empty_index = new_idx
        is_empty_idx = new_idx
        print(f"[clean_graphs.py] {file_name}: Added missing is_empty column at index {new_idx}")

    # 3. Print feature presence/absence by index
    all_possible_feats = sorted(set([k for k in feature_indices.keys()] + ['centroid_x', 'centroid_y', 'is_empty']))
    present_feats = []
    missing_feats = []
    for feat in all_possible_feats:
        if feat == 'is_empty':
            if is_empty_idx is not None and is_empty_idx < x.shape[1]:
                present_feats.append(feat)
            else:
                missing_feats.append(feat)
        elif feat in feature_indices and feature_indices[feat] < x.shape[1]:
            present_feats.append(feat)
        else:
            missing_feats.append(feat)
    print(f"[clean_graphs.py] {file_name}: Present features: {present_feats}, Missing features: {missing_feats}")

    # 4. Rebuild the feature matrix column by column
    x_final_cols = []
    amenity_col_idx = 0
    for col in range(x.shape[1]):
        if col == is_empty_idx:
            x_final_cols.append(x[:, col:col+1])
        else:
            feat_val = x[:, col:col+1]
            feat_log = torch.log1p(torch.clamp(feat_val, min=0.0))
            f_min = global_min[amenity_col_idx]
            f_max = global_max[amenity_col_idx]
            feat_norm = (feat_log - f_min) / (f_max - f_min + eps)
            x_final_cols.append(feat_norm)
            amenity_col_idx += 1

    x_processed = torch.cat(x_final_cols, dim=1)

    # --- Pad or truncate to expected_dim ---
    current_dim = x_processed.shape[1]
    if current_dim < expected_dim:
        diff = expected_dim - current_dim
        pad = torch.zeros((x_processed.shape[0], diff), device=x_processed.device)
        x_processed = torch.cat([x_processed, pad], dim=1)
        print(f"[clean_graphs.py] Padded {file_name} from {current_dim} to {expected_dim}")
    elif current_dim > expected_dim:
        x_processed = x_processed[:, :expected_dim]
        print(f"[clean_graphs.py] Truncated {file_name} from {current_dim} to {expected_dim}")

    # 3. Add H3 Resolution (7, 8, 9 -> 0.0, 0.5, 1.0)
    res_val = (get_res(file_name) - 7) / 2.0
    res_col = torch.full((x_processed.shape[0], 1), res_val)
    data.x = torch.cat([x_processed, res_col], dim=1)


    # Safety: Final check for NaNs before saving
    data.x = torch.nan_to_num(data.x, nan=0.0)

    # Remove fill_mask attribute if present
    if hasattr(data, 'fill_mask'):
        delattr(data, 'fill_mask')
        print(f"[clean_graphs.py] {file_name}: Removed fill_mask attribute.")

    torch.save(data, os.path.join(output_dir, file_name))

print("Done. is_empty remains binary. Amenities are log-normalized. Resolution added.")