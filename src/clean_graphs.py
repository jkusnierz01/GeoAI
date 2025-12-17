import os
import torch
import glob


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

pt_files = glob.glob(os.path.join(input_dir, "*.pt"))
print(f"Found {len(pt_files)} files to process.")

# First pass: compute global min and max for each feature (except is_empty)
global_min = None
global_max = None
feature_dim = None
is_empty_index = None

for file_path in pt_files:
    try:
        data = torch.load(file_path, weights_only=False)
        # Find is_empty_index if present
        if hasattr(data, 'is_empty_index'):
            is_empty_index = int(getattr(data, 'is_empty_index'))
        # Remove centroid_x and centroid_y features and their indices
        for feat in ['centroid_x', 'centroid_y']:
            idx_attr = f'{feat}_index'
            if hasattr(data, idx_attr):
                idx = int(getattr(data, idx_attr))
                if hasattr(data, 'x') and data.x.shape[1] > idx:
                    data.x = torch.cat([data.x[:, :idx], data.x[:, idx+1:]], dim=1)
                delattr(data, idx_attr)
        # Now update global min/max (excluding is_empty column)
        if hasattr(data, 'x') and data.x is not None:
            x = data.x
            if is_empty_index is not None:
                mask = [i for i in range(x.shape[1]) if i != is_empty_index]
                x_no_is_empty = x[:, mask]
            else:
                x_no_is_empty = x
            if global_min is None:
                global_min = x_no_is_empty.min(dim=0).values
                global_max = x_no_is_empty.max(dim=0).values
                feature_dim = x.shape[1]
            else:
                global_min = torch.minimum(global_min, x_no_is_empty.min(dim=0).values)
                global_max = torch.maximum(global_max, x_no_is_empty.max(dim=0).values)
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

        # Remove centroid_x and centroid_y features and their indices
        for feat in ['centroid_x', 'centroid_y']:
            idx_attr = f'{feat}_index'
            if hasattr(data, idx_attr):
                idx = int(getattr(data, idx_attr))
                if hasattr(data, 'x') and data.x.shape[1] > idx:
                    data.x = torch.cat([data.x[:, :idx], data.x[:, idx+1:]], dim=1)
                    print(f"Removed feature column '{feat}' at index {idx} from {file_name}")
                    changed = True
                delattr(data, idx_attr)
                changed = True

        # Remove other metadata if present
        for attr in ['has_is_empty', 'mapping_log']:
            if hasattr(data, attr):
                delattr(data, attr)
                changed = True

        # Normalize features except is_empty
        if hasattr(data, 'x') and data.x is not None and global_min is not None and global_max is not None:
            x = data.x
            if hasattr(data, 'is_empty_index'):
                is_empty_index = int(getattr(data, 'is_empty_index'))
            else:
                is_empty_index = None
            if x.shape[1] == feature_dim:
                mask = [i for i in range(x.shape[1]) if i != is_empty_index]
                x_to_norm = x[:, mask]
                min_vals = global_min.unsqueeze(0)
                max_vals = global_max.unsqueeze(0)
                denom = (max_vals - min_vals)
                denom[denom == 0] = 1
                x_norm = (x_to_norm - min_vals) / denom
                # Reconstruct x with is_empty column unchanged
                if is_empty_index is not None:
                    x_new = []
                    for i in range(x.shape[1]):
                        if i == is_empty_index:
                            x_new.append(x[:, i:i+1])
                        else:
                            x_new.append(x_norm[:, 0:1])
                            x_norm = x_norm[:, 1:]
                    data.x = torch.cat(x_new, dim=1)
                else:
                    data.x = x_norm
                print(f"Normalized features for {file_name} (except is_empty)")
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
