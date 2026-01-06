import os
import torch
import glob
import re

# input_dir = "dataset"
# output_dir = "dataset_aligned"

# input_dir = "src/benchmarks/graphs_old"
# output_dir = "src/benchmarks/graphs"

input_dir = "src/benchmarks/graph_for_satelite_old"
output_dir = "src/benchmarks/graph_for_satelite"

os.makedirs(output_dir, exist_ok=True)

pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
print(f"Found {len(pt_files)} files.")

print("Processing graphs (Removing centroids + Log1p on Counts only)...")

for file_path in pt_files:
    data = torch.load(file_path, weights_only=False)
    file_name = os.path.basename(file_path)
    x = data.x.float()

    indices_to_remove = []
    for feat in ['centroid_x', 'centroid_y']:
        idx_attr = f'{feat}_index'
        if hasattr(data, idx_attr):
            indices_to_remove.append(int(getattr(data, idx_attr)))
            delattr(data, idx_attr)
            
    is_empty_idx = int(data.is_empty_index) if hasattr(data, 'is_empty_index') else None
    
    num_feats = x.shape[1]
    
    keep_mask = torch.ones(num_feats, dtype=torch.bool, device=x.device)
    keep_mask[indices_to_remove] = False
    
    log_mask = keep_mask.clone()
    if is_empty_idx is not None and is_empty_idx < num_feats:
        log_mask[is_empty_idx] = False

    x_processed = torch.where(log_mask.unsqueeze(0), torch.log1p(x), x)

    x_final = x_processed[:, keep_mask]

    if is_empty_idx is not None and is_empty_idx < num_feats:
        if is_empty_idx in indices_to_remove:
            is_empty_idx = None 
        else:
            shift = sum(1 for i in indices_to_remove if i < is_empty_idx)
            new_idx = is_empty_idx - shift
            data.is_empty_index = new_idx
            is_empty_idx = new_idx 

    if is_empty_idx is None:
        is_empty_col = torch.zeros((x_final.shape[0], 1), dtype=x_final.dtype, device=x_final.device)
        x_final = torch.cat([x_final, is_empty_col], dim=1)
        data.is_empty_index = x_final.shape[1] - 1

    data.x = torch.nan_to_num(x_final, nan=0.0)

    if hasattr(data, 'fill_mask'):
        delattr(data, 'fill_mask')

    torch.save(data, os.path.join(output_dir, file_name))

print("Done.")