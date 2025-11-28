import torch
from torchvision import transforms
from transformers import AutoModel
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# ---------------------------
# Config
# ---------------------------
IMAGE_DIR = "test"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
model = AutoModel.from_pretrained(model_name, device_map="auto")
model.eval()
print(f"Model loaded on: {device}")

# Transform
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------
# Functions
# ---------------------------
def get_embedding(image_path):
    """Get normalized embedding for an image using DINO v3."""
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    inputs = {"pixel_values": img_tensor}
    with torch.inference_mode():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return F.normalize(embedding, p=2, dim=-1)

def cosine_sim(emb1, emb2):
    """Compute cosine similarity."""
    return torch.mm(emb1, emb2.t()).item()

# ---------------------------
# Load images
# ---------------------------
image_paths = sorted([
    os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f"Found {len(image_paths)} images:")
for p in image_paths:
    print(f"  - {p}")

# ---------------------------
# Compute embeddings
# ---------------------------
embeddings = {}
for path in image_paths:
    embeddings[path] = get_embedding(path)
    print(f"Computed embedding for: {os.path.basename(path)}")

# ---------------------------
# Compute pairwise similarities
# ---------------------------
pairs = []
for i, img1 in enumerate(image_paths):
    for j, img2 in enumerate(image_paths):
        if i < j:
            sim = cosine_sim(embeddings[img1], embeddings[img2])
            pairs.append((img1, img2, sim))

pairs.sort(key=lambda x: x[2], reverse=True)
print(f"Total pairs: {len(pairs)}")

# ---------------------------
# Save each pair as a separate plot
# ---------------------------
for idx, (img1_path, img2_path, score) in enumerate(pairs, 1):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Left image
    axes[0].imshow(img1)
    axes[0].set_title(os.path.basename(img1_path), fontsize=12)
    axes[0].axis('off')

    # Right image
    axes[1].imshow(img2)
    axes[1].set_title(os.path.basename(img2_path), fontsize=12)
    axes[1].axis('off')

    # Similarity label
    if score >= 0.8:
        color = 'green'
        label = 'Very Similar'
    elif score >= 0.5:
        color = 'orange'
        label = 'Similar'
    else:
        color = 'red'
        label = 'Different'

    fig.suptitle(f'Cosine Similarity: {score:.4f} ({label})',
                 fontsize=16, color=color, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(OUTPUT_DIR, f'pair_{idx}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)  # Close figure to save memory
    print(f"Saved plot: {plot_path}")

print("All plots saved successfully.")
