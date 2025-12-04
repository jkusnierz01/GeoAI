import json
import numpy as np

# Load results
with open("scale_similarity_results.json", "r") as f:
    data = json.load(f)

same = data["same_region_scaled"]     # list of (filename, score)
random_scaled = data["random_scaled"] # list of (f1, f2, score)

stats = {}

# SAME region scale stats
stats["scale_same"] = [s for (_, s) in same]

# RANDOM region scale stats
stats["scale_random"] = [s for (_, _, s) in random_scaled]


# Print formatted statistics
for key in sorted(stats.keys()):
    vals = np.array(stats[key], dtype=float)
    n = len(vals)
    mean = vals.mean()
    std = vals.std()
    vmin = vals.min()
    vmax = vals.max()

    print(f"{key}: n={n}, mean={mean:.4f}, std={std:.4f}, min={vmin:.4f}, max={vmax:.4f}")
