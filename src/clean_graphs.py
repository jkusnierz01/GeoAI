import os
import torch
import glob
import re

# input_dir = "dataset"
# output_dir = "dataset_aligned"

input_dir = "src/benchmarks/graphs_old"
output_dir = "src/benchmarks/graphs"

os.makedirs(output_dir, exist_ok=True)

pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
print(f"Found {len(pt_files)} files.")

print("Processing graphs (Removing centroids + Log1p on Counts only)...")

for file_path in pt_files:
    data = torch.load(file_path, weights_only=False)
    file_name = os.path.basename(file_path)
    x = data.x.float()

    # --- 1. Identify Special Columns (to exclude from Log1p) ---
    
    # Track indices to remove (centroids)
    indices_to_remove = []
    for feat in ['centroid_x', 'centroid_y']:
        idx_attr = f'{feat}_index'
        if hasattr(data, idx_attr):
            indices_to_remove.append(int(getattr(data, idx_attr)))
            delattr(data, idx_attr)
            
    # Track index of is_empty (to preserve)
    is_empty_idx = int(data.is_empty_index) if hasattr(data, 'is_empty_index') else None
    
    # --- 2. Construct Masks ---
    
    num_feats = x.shape[1]
    
    # Mask for columns to KEEP in the final output
    keep_mask = torch.ones(num_feats, dtype=torch.bool, device=x.device)
    keep_mask[indices_to_remove] = False
    
    # Mask for columns to Apply Log1p (Keep AND Not is_empty)
    log_mask = keep_mask.clone()
    if is_empty_idx is not None and is_empty_idx < num_feats:
        log_mask[is_empty_idx] = False

    # --- 3. Process Data ---
    x_processed = torch.where(log_mask.unsqueeze(0), torch.log1p(x), x)

    x_final = x_processed[:, keep_mask]

    if is_empty_idx is not None and is_empty_idx < num_feats:
        # Check if we accidentally deleted it (unlikely, but safety first)
        if is_empty_idx in indices_to_remove:
            # If it was deleted, we must recreate it below
            is_empty_idx = None 
        else:
            # Calculate shift
            shift = sum(1 for i in indices_to_remove if i < is_empty_idx)
            new_idx = is_empty_idx - shift
            data.is_empty_index = new_idx
            is_empty_idx = new_idx # Update var for check below

    # If is_empty didn't exist (or was deleted), create it now
    if is_empty_idx is None:
        is_empty_col = torch.zeros((x_final.shape[0], 1), dtype=x_final.dtype, device=x_final.device)
        x_final = torch.cat([x_final, is_empty_col], dim=1)
        data.is_empty_index = x_final.shape[1] - 1

    # --- 5. Save ---
    
    data.x = torch.nan_to_num(x_final, nan=0.0)

    if hasattr(data, 'fill_mask'):
        delattr(data, 'fill_mask')

    torch.save(data, os.path.join(output_dir, file_name))

print("Done.")