import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import rasterio
from torchvision import transforms
from transformers import AutoModel
import json

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
IMAGE_ROOT = "images"
FOLDERS = ["clean", "cloud", "winter"]

model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------
# LOAD MODEL
# ---------------------------------------------
model = AutoModel.from_pretrained(model_name, device_map="auto")
model.eval()

# ---------------------------------------------
# TRANSFORM FOR SENTINEL2 TIFF
# ---------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------
# TIFF READER
# ---------------------------------------------
def load_tiff_rgb(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(float)

    if arr.shape[0] >= 3:
        rgb = np.stack([arr[2], arr[1], arr[0]], axis=2)
    else:
        raise ValueError(f"TIFF has only {arr.shape[0]} bands: {path}")

    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return rgb

# ---------------------------------------------
# EMBEDDINGS
# ---------------------------------------------
def get_embedding(path):
    rgb = load_tiff_rgb(path)
    img = transform(rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model(pixel_values=img)
        emb = out.last_hidden_state[:, 0, :]
    return F.normalize(emb, p=2, dim=-1)

def cosine(a, b):
    return torch.mm(a, b.t()).item()

# ---------------------------------------------
# UTILITY: extract region-id from filename
# ---------------------------------------------
def get_region_id(filename):
    """
    Example:
    '87118134affffff_sentinel2.tif' -> '87118134affffff'
    """
    return filename.split("_")[0]

# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
    folder_files = {}
    id_to_files = {name: {} for name in FOLDERS}

    # -----------------------------------------
    # SCAN DIRECTORIES AND MAP REGION IDs
    # -----------------------------------------
    for name in FOLDERS:
        files = [f for f in os.listdir(os.path.join(IMAGE_ROOT, name))
                 if f.endswith(".tiff") or f.endswith(".tif")]
        folder_files[name] = files

        for f in files:
            region_id = get_region_id(f)
            id_to_files[name][region_id] = f

    # -----------------------------------------
    # FIND COMMON REGION IDs
    # -----------------------------------------
    common_ids = set(id_to_files[FOLDERS[0]].keys())
    for name in FOLDERS[1:]:
        common_ids &= set(id_to_files[name].keys())

    print(f"Common region IDs across folders: {len(common_ids)}")

    pairs = [("clean", "cloud"), ("clean", "winter"), ("cloud", "winter")]

    results_same_region = []
    results_random = []

    # -----------------------------------------
    # SAME REGION COMPARISONS
    # -----------------------------------------
    for region in sorted(common_ids):
        for A, B in pairs:
            fA = id_to_files[A][region]
            fB = id_to_files[B][region]

            path_a = os.path.join(IMAGE_ROOT, A, fA)
            path_b = os.path.join(IMAGE_ROOT, B, fB)

            emb_a = get_embedding(path_a)
            emb_b = get_embedding(path_b)

            score = cosine(emb_a, emb_b)
            results_same_region.append((A, B, fA, fB, score))
            print(f"[SAME] {A}-{B} ({region}): {score:.4f}")

    # -----------------------------------------
    # RANDOM REGION COMPARISONS
    # -----------------------------------------
    for A, B in pairs:
        A_files = folder_files[A]
        B_files = folder_files[B]

        for _ in range(len(common_ids)):
            fa = np.random.choice(A_files)
            fb = np.random.choice(B_files)

            emb_a = get_embedding(os.path.join(IMAGE_ROOT, A, fa))
            emb_b = get_embedding(os.path.join(IMAGE_ROOT, B, fb))

            score = cosine(emb_a, emb_b)
            results_random.append((A, B, fa, fb, score))
            print(f"[RANDOM] {A}-{B} {fa} vs {fb}: {score:.4f}")

    # -----------------------------------------
    # SAVE RESULTS
    # -----------------------------------------
    out = {
        "same_region": results_same_region,
        "random_pairs": results_random
    }

    with open("condition_similarity_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved condition_similarity_results.json")

if __name__ == "__main__":
    main()
