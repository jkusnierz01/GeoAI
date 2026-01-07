import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import pickle
import rasterio

# ==========================================
# 1. Helper: Load Embeddings (Optional)
# ==========================================
def load_embedding_map(embedding_path, key_col='h3_index'):
    if not embedding_path or not os.path.exists(embedding_path):
        print("Warning: Embedding path invalid or not provided.")
        return {}, 0

    print(f"Loading external embeddings from: {embedding_path}")
    
    if embedding_path.endswith('.pkl'):
        obj = pd.read_pickle(embedding_path)
        if isinstance(obj, dict):
            embedding_map = {str(k): np.array(v, dtype=np.float32) for k, v in obj.items()}
            emb_dim = len(next(iter(embedding_map.values())))
            print(f"Loaded embeddings for {len(embedding_map)} locations. Vector dim: {emb_dim}")
            return embedding_map, emb_dim
        else:
            df_emb = obj
    elif embedding_path.endswith('.csv'):
        df_emb = pd.read_csv(embedding_path)
    else:
        raise ValueError("Unsupported file format. Use .pkl or .csv")

    if key_col in df_emb.columns:
        df_emb = df_emb.set_index(key_col)
    
    df_emb = df_emb.select_dtypes(include=[np.number])
    embedding_map = {str(k): v.values.astype(np.float32) for k, v in df_emb.iterrows()}
    
    if not embedding_map:
         return {}, 0

    emb_dim = len(next(iter(embedding_map.values())))
    print(f"Loaded embeddings for {len(embedding_map)} locations. Vector dim: {emb_dim}")
    return embedding_map, emb_dim

# ==========================================
# 2. Dataset Class
# ==========================================
class EuroSATDataset(Dataset):
    def __init__(self, df, root_dir, embedding_map=None, embedding_dim=0, transform=None, class_encoder=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.class_encoder = class_encoder
        self.embedding_map = embedding_map
        self.embedding_dim = embedding_dim
        self.use_embeddings = (embedding_map is not None and len(embedding_map) > 0)
        
        if self.use_embeddings:
            self.zero_embedding = np.zeros(embedding_dim, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['class'], row['filename'])
        
        image = None
        try:
            if img_path.lower().endswith('.tif') or img_path.lower().endswith('.tiff'):
                with rasterio.open(img_path) as src:
                    # EuroSAT MS Bands: B04(Red), B03(Green), B02(Blue)
                    r = src.read(4)
                    g = src.read(3)
                    b = src.read(2)
                    img_array = np.dstack((r, g, b))
                    # Normalize uint16 to uint8
                    img_array = (np.clip(img_array / 3000.0, 0, 1) * 255).astype(np.uint8)
                    image = Image.fromarray(img_array)
            else:
                image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (64, 64))

        if self.transform:
            image = self.transform(image)

        label = self.class_encoder.transform([row['class']])[0]

        if self.use_embeddings:
            h3_idx = str(row['h3_index'])
            emb_vector = self.embedding_map.get(h3_idx, self.zero_embedding)
            return image, emb_vector, label
        else:
            return image, torch.tensor(0.0), label

# ==========================================
# 3. Flexible Model
# ==========================================
class FlexibleModel(nn.Module):
    def __init__(self, num_classes, use_embeddings=False, input_embedding_dim=0):
        super(FlexibleModel, self).__init__()
        self.use_embeddings = use_embeddings

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_out_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity() 

        if self.use_embeddings:
            self.loc_fc = nn.Sequential(
                nn.Linear(input_embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            fusion_dim = self.cnn_out_dim + 64
        else:
            fusion_dim = self.cnn_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, emb_vector=None):
        img_feat = self.cnn(x)
        
        if self.use_embeddings and emb_vector is not None:
            loc_feat = self.loc_fc(emb_vector)
            combined = torch.cat((img_feat, loc_feat), dim=1)
        else:
            combined = img_feat
        
        return self.classifier(combined)

# ==========================================
# 4. Training & Evaluation Engine
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, use_embeddings):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, vectors, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        if use_embeddings:
            vectors = vectors.float().to(device)
            outputs = model(images, vectors)
        else:
            outputs = model(images, None)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(loader), accuracy_score(all_labels, all_preds)

def evaluate(model, loader, criterion, device, use_embeddings):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, vectors, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if use_embeddings:
                vectors = vectors.float().to(device)
                outputs = model(images, vectors)
            else:
                outputs = model(images, None)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), accuracy_score(all_labels, all_preds), all_labels, all_preds

# ==========================================
# 5. Main Execution
# ==========================================
def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.csv_file):
        print("Error: CSV file not found.")
        return
        
    df = pd.read_csv(args.csv_file)
    le_class = LabelEncoder()
    df['class_encoded'] = le_class.fit_transform(df['class'])
    num_classes = len(le_class.classes_)

    # --- Load Embeddings ---
    embedding_map = {}
    emb_dim = 0
    use_embeddings = False
    if args.embedding_path:
        embedding_map, emb_dim = load_embedding_map(args.embedding_path, key_col='h3_index')
        if emb_dim > 0:
            use_embeddings = True
    
    print(f"Mode: {'GEO-AWARE (Embeddings)' if use_embeddings else 'BASELINE (Image Only)'}")

    # --- Data Splits (Train/Val/Test) ---
    # 1. Split into (Train+Val) and Test (80/20)
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
    
    # 2. Split (Train+Val) into Train and Val (80/20 of the remaining)
    # Result approx: Train 64%, Val 16%, Test 20%
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['class'])

    print(f"Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # --- Transforms ---
    t_train = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t_eval = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- Create Datasets ---
    train_set = EuroSATDataset(train_df, args.data_root, embedding_map, emb_dim, t_train, le_class)
    val_set = EuroSATDataset(val_df, args.data_root, embedding_map, emb_dim, t_eval, le_class)
    test_set = EuroSATDataset(test_df, args.data_root, embedding_map, emb_dim, t_eval, le_class)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Model Setup ---
    model = FlexibleModel(
        num_classes=num_classes, 
        use_embeddings=use_embeddings, 
        input_embedding_dim=emb_dim
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print("\nStarting Training...")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Train on TRAIN set
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_embeddings)
        
        # Evaluate on VAL set
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device, use_embeddings)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {t_acc:.2%} Loss: {t_loss:.4f} | Val Acc: {v_acc:.2%} Loss: {v_loss:.4f}")

        # Optional: Save best model state
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            # torch.save(model.state_dict(), "best_model.pth") 

    # --- Final Test Evaluation ---
    print("\n" + "="*30)
    print("FINAL TEST REPORT")
    print("="*30)
    
    test_loss, test_acc, true_labels, pred_labels = evaluate(model, test_loader, criterion, device, use_embeddings)
    
    # Calculate detailed metrics
    report = classification_report(true_labels, pred_labels, target_names=le_class.classes_, digits=4)
    
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test Loss:     {test_loss:.4f}\n")
    print("Detailed Classification Report:")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="eurosat_h3_index.csv", help="Main CSV with filenames and H3 indexes")
    parser.add_argument("--data_root", type=str, default="EuroSAT_MS", help="Image folder")
    parser.add_argument("--embedding_path", type=str, default=None, help="Path to .pkl/.csv with embeddings (optional)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    run_experiment(args)