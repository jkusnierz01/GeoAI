import os
import torch
import glob

# input_dir = os.path.join("dataset")
# output_dir = os.path.join("dataset_aligned")

input_dir = os.path.join("src/benchmarks/graphs_old")
output_dir = os.path.join("src/benchmarks/graphs")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
print(f"Found {len(pt_files)} files to process.")

# First pass: compute global min and max for each feature
global_min = None
global_max = None
feature_dim = None

for file_path in pt_files:
    try:
        data = torch.load(file_path, weights_only=False)
        # Remove specified attributes and columns to get correct features
        for attr in ['is_empty', 'centroid_x', 'centroid_y']:
            if hasattr(data, attr):
                delattr(data, attr)
        feature_indices = {}
        for feat in ['is_empty', 'centroid_x', 'centroid_y']:
            idx_attr = f'{feat}_index'
            if hasattr(data, idx_attr):
                feature_indices[feat] = int(getattr(data, idx_attr))
                delattr(data, idx_attr)
        if hasattr(data, 'x') and feature_indices:
            for feat, idx in sorted(feature_indices.items(), key=lambda x: -x[1]):
                if data.x.shape[1] > idx:
                    data.x = torch.cat([data.x[:, :idx], data.x[:, idx+1:]], dim=1)
        # Now update global min/max
        if hasattr(data, 'x') and data.x is not None:
            x = data.x
            if global_min is None:
                global_min = x.min(dim=0).values
                global_max = x.max(dim=0).values
                feature_dim = x.shape[1]
            else:
                global_min = torch.minimum(global_min, x.min(dim=0).values)
                global_max = torch.maximum(global_max, x.max(dim=0).values)
    except Exception as e:
        print(f"Error processing {file_path} in min/max pass: {e}")

if global_min is None or global_max is None:
    print("No features found to normalize.")
else:
    print(f"Global min: {global_min}")
    print(f"Global max: {global_max}")

# Second pass: clean and normalize
for file_path in pt_files:
    try:
        data = torch.load(file_path, weights_only=False)
        file_name = os.path.basename(file_path)
        changed = False

        # Remove specified attributes if present
        for attr in ['is_empty', 'centroid_x', 'centroid_y']:
            if hasattr(data, attr):
                delattr(data, attr)
                changed = True

        # Remove feature columns if present
        feature_indices = {}
        for feat in ['is_empty', 'centroid_x', 'centroid_y']:
            idx_attr = f'{feat}_index'
            if hasattr(data, idx_attr):
                feature_indices[feat] = int(getattr(data, idx_attr))
                delattr(data, idx_attr)
                changed = True
        if hasattr(data, 'x') and feature_indices:
            for feat, idx in sorted(feature_indices.items(), key=lambda x: -x[1]):
                if data.x.shape[1] > idx:
                    data.x = torch.cat([data.x[:, :idx], data.x[:, idx+1:]], dim=1)
                    print(f"Removed feature column '{feat}' at index {idx} from {file_name}")
                    changed = True

        # Remove other metadata if present
        for attr in ['has_is_empty', 'mapping_log']:
            if hasattr(data, attr):
                delattr(data, attr)
                changed = True

        # Normalize features if present
        if hasattr(data, 'x') and data.x is not None and global_min is not None and global_max is not None:
            x = data.x
            if x.shape[1] == feature_dim:
                min_vals = global_min.unsqueeze(0)
                max_vals = global_max.unsqueeze(0)
                denom = (max_vals - min_vals)
                denom[denom == 0] = 1
                data.x = (x - min_vals) / denom
                print(f"Normalized features for {file_name}")
                changed = True
            else:
                print(f"Warning: Feature dimension mismatch for {file_name}, skipping normalization.")

        if changed:
            out_path = os.path.join(output_dir, file_name)
            torch.save(data, out_path)
            print(f"Cleaned and saved {file_name}")
        else:
            print(f"No changes for {file_name}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("Done processing all graphs.")
