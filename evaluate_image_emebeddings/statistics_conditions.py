import json
import numpy as np

# Load JSON
with open("condition_similarity_results.json", "r") as f:
    data = json.load(f)

same = data["same_region"]
random_pairs = data["random_pairs"]

# Collect scores by category
stats = {}

# SAME REGION PAIRS
for A, B, fA, fB, score in same:
    key = f"{A}-{B}"
    if key not in stats:
        stats[key] = []
    stats[key].append(score)

# RANDOM REGION PAIRS
for A, B, fA, fB, score in random_pairs:
    key = f"random_{A}-{B}"
    if key not in stats:
        stats[key] = []
    stats[key].append(score)

# Print summary
for key in sorted(stats.keys()):
    vals = np.array(stats[key], dtype=float)
    n = len(vals)
    mean = vals.mean()
    std = vals.std()
    vmin = vals.min()
    vmax = vals.max()

    print(f"{key}: n={n}, mean={mean:.4f}, std={std:.4f}, min={vmin:.4f}, max={vmax:.4f}")
    with open("conditions_stats.txt", "a") as out_f:
        out_f.write(f"{key}: n={n}, mean={mean:.4f}, std={std:.4f}, min={vmin:.4f}, max={vmax:.4f}\n")
