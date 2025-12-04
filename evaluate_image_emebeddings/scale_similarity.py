import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import rasterio
from torchvision import transforms
from transformers import AutoModel

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
IMAGE_DIR = "images/clean"
CROP_SIZE = 128          # size of small crop
NUM_RANDOM = 50          # number of random pairs

model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(model_name, device_map="auto")
model.eval()


# ---------------------------------------------
# TRANSFORMS
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
# TIFF LOADER
# ---------------------------------------------
def load_tiff_rgb(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(float)
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=2)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return rgb


# ---------------------------------------------
# EMBEDDING
# ---------------------------------------------
def get_embedding_rgb(rgb):
    img = transform(rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model(pixel_values=img)
        emb = out.last_hidden_state[:, 0, :]
    return F.normalize(emb, p=2, dim=-1)


def cosine(a, b):
    return torch.mm(a, b.t()).item()


# ---------------------------------------------
# CROP FUNCTION
# ---------------------------------------------
def center_crop(rgb, size):
    h, w, _ = rgb.shape
    ch = size
    cw = size
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return rgb[y0:y0+ch, x0:x0+cw]


def random_crop(rgb, size):
    h, w, _ = rgb.shape
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    return rgb[y:y+size, x:x+size]


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".tif") or f.endswith(".tiff")]
    print(f"Found {len(files)} clean images")

    results_same = []
    results_random = []

    for fname in files:
        path = os.path.join(IMAGE_DIR, fname)
        rgb = load_tiff_rgb(path)

        # Original embedding
        emb_orig = get_embedding_rgb(rgb)

        # Center crop
        crop = center_crop(rgb, CROP_SIZE)
        emb_crop = get_embedding_rgb(crop)

        score = cosine(emb_orig, emb_crop)
        results_same.append((fname, score))
        print(f"[SCALE SAME] {fname}: {score:.4f}")

    # -----------------------------------------
    # RANDOM REGION SCALE COMPARISON
    # -----------------------------------------
    for _ in range(NUM_RANDOM):
        f1, f2 = np.random.choice(files, 2, replace=True)

        rgb1 = load_tiff_rgb(os.path.join(IMAGE_DIR, f1))
        rgb2 = load_tiff_rgb(os.path.join(IMAGE_DIR, f2))

        crop1 = center_crop(rgb1, CROP_SIZE)
        crop2 = center_crop(rgb2, CROP_SIZE)

        emb1 = get_embedding_rgb(crop1)
        emb2 = get_embedding_rgb(crop2)

        score = cosine(emb1, emb2)
        results_random.append((f1, f2, score))
        print(f"[SCALE RANDOM] {f1} vs {f2}: {score:.4f}")

    # Save
    import json
    out = {
        "same_region_scaled": results_same,
        "random_scaled": results_random
    }
    with open("scale_similarity_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved scale_similarity_results.json")


if __name__ == "__main__":
    main()
