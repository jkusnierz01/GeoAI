import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
from tqdm import tqdm
import rasterio

# ==========================================
# 1. Helper: Load Embeddings (Strict)
# ==========================================
def load_embedding_map(embedding_path, key_col='h3_index'):
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    print(f"Loading external embeddings from: {embedding_path}")
    
    if embedding_path.endswith('.pkl'):
        obj = pd.read_pickle(embedding_path)
        if isinstance(obj, dict):
            embedding_map = {str(k): np.array(v, dtype=np.float32) for k, v in obj.items()}
        else:
            df_emb = obj
            if key_col in df_emb.columns:
                df_emb = df_emb.set_index(key_col)
            df_emb = df_emb.select_dtypes(include=[np.number])
            embedding_map = {str(k): v.values.astype(np.float32) for k, v in df_emb.iterrows()}
    elif embedding_path.endswith('.csv'):
        df_emb = pd.read_csv(embedding_path)
        if key_col in df_emb.columns:
            df_emb = df_emb.set_index(key_col)
        df_emb = df_emb.select_dtypes(include=[np.number])
        embedding_map = {str(k): v.values.astype(np.float32) for k, v in df_emb.iterrows()}
    else:
        raise ValueError("Unsupported file format. Use .pkl or .csv")

    if not embedding_map:
        raise ValueError("Embedding map is empty. Check your data.")

    emb_dim = len(next(iter(embedding_map.values())))
    print(f"Loaded embeddings for {len(embedding_map)} locations. Vector dim: {emb_dim}")
    return embedding_map, emb_dim

# ==========================================
# 2. Dataset Class
# ==========================================
class EuroSATDataset(Dataset):
    def __init__(self, df, root_dir, embedding_map, embedding_dim, transform=None, class_encoder=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.class_encoder = class_encoder
        self.embedding_map = embedding_map
        self.embedding_dim = embedding_dim
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
                    r = src.read(4)
                    g = src.read(3)
                    b = src.read(2)
                    img_array = np.dstack((r, g, b))
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

        h3_idx = str(row['h3_index'])
        emb_vector = self.embedding_map.get(h3_idx, self.zero_embedding)
        
        return image, emb_vector, label

# ==========================================
# 3. Models and Fusion Architectures
# ==========================================
class CrossModalAttention(nn.Module):
    def __init__(self, img_dim, graph_dim, hidden_dim=256):
        super(CrossModalAttention, self).__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.graph_proj = nn.Sequential(nn.Linear(graph_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2), 
            nn.Softmax(dim=1)         
        )

    def forward(self, img_feat, graph_feat):
        h_img = self.img_proj(img_feat)
        h_graph = self.graph_proj(graph_feat)
        joint_features = torch.cat([h_img, h_graph], dim=1)
        attn_weights = self.attn_net(joint_features)
        
        alpha_img, alpha_graph = attn_weights[:, 0].unsqueeze(1), attn_weights[:, 1].unsqueeze(1)
        return (alpha_img * h_img) + (alpha_graph * h_graph)

class TransformerCrossAttention(nn.Module):
    def __init__(self, img_dim, graph_dim, hidden_dim=256, num_heads=4):
        super(TransformerCrossAttention, self).__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        
        # Batch_first=True expects shape (Batch, Sequence, Features)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, img_feat, graph_feat):
        # We treat the vectors as sequences of length 1
        # Query = Graph context, Key/Value = Image visual features
        query = self.graph_proj(graph_feat).unsqueeze(1) # (Batch, 1, hidden_dim)
        key_value = self.img_proj(img_feat).unsqueeze(1) # (Batch, 1, hidden_dim)
        
        attn_out, _ = self.mha(query, key_value, key_value)
        attn_out = attn_out.squeeze(1) # Back to (Batch, hidden_dim)
        
        # Add residual connection and normalize
        return self.norm(attn_out + query.squeeze(1))

class BilinearPooling(nn.Module):
    def __init__(self, img_dim, graph_dim, output_dim=256):
        super(BilinearPooling, self).__init__()
        # Outer product dimension
        self.bilinear_dim = img_dim * graph_dim 
        
        self.proj = nn.Sequential(
            nn.Linear(self.bilinear_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, img_feat, graph_feat):
        # Compute outer product: (B, img_dim, 1) x (B, 1, graph_dim) -> (B, img_dim, graph_dim)
        outer_prod = torch.bmm(img_feat.unsqueeze(2), graph_feat.unsqueeze(1))
        outer_prod = outer_prod.view(outer_prod.size(0), -1) # Flatten
        
        # Standard Bilinear Pooling stabilization: Signed Square Root + L2 Norm
        outer_prod = torch.sign(outer_prod) * torch.sqrt(torch.abs(outer_prod) + 1e-9)
        outer_prod = torch.nn.functional.normalize(outer_prod, p=2, dim=1)
        
        return self.proj(outer_prod)

class MultimodalEuroSATModel(nn.Module):
    def __init__(self, num_classes, input_embedding_dim, fusion_method='concat'):
        super(MultimodalEuroSATModel, self).__init__()
        self.fusion_method = fusion_method.lower()
        
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_out_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity() 

        self.graph_out_dim = 64
        self.loc_fc = nn.Sequential(
            nn.Linear(input_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.graph_out_dim)
        )

        # Map fusion choices
        self.hidden_fusion_dim = 256
        if self.fusion_method == 'attention':
            self.fusion_module = CrossModalAttention(self.cnn_out_dim, self.graph_out_dim, self.hidden_fusion_dim)
            classifier_in_dim = self.hidden_fusion_dim
        elif self.fusion_method == 'transformer':
            self.fusion_module = TransformerCrossAttention(self.cnn_out_dim, self.graph_out_dim, self.hidden_fusion_dim)
            classifier_in_dim = self.hidden_fusion_dim
        elif self.fusion_method == 'bilinear':
            self.fusion_module = BilinearPooling(self.cnn_out_dim, self.graph_out_dim, self.hidden_fusion_dim)
            classifier_in_dim = self.hidden_fusion_dim
        elif self.fusion_method == 'concat':
            classifier_in_dim = self.cnn_out_dim + self.graph_out_dim
        else:
            raise ValueError("fusion_method must be: 'concat', 'attention', 'transformer', or 'bilinear'")

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, emb_vector):
        img_feat = self.cnn(x)
        loc_feat = self.loc_fc(emb_vector)
        
        if self.fusion_method == 'concat':
            combined = torch.cat((img_feat, loc_feat), dim=1)
        else:
            combined = self.fusion_module(img_feat, loc_feat)
            
        return self.classifier(combined)

# ==========================================
# 4. Training & Evaluation Engine
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, vectors, labels in pbar:
        images, vectors, labels = images.to(device), vectors.float().to(device), labels.to(device)
        outputs = model(images, vectors)

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

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, vectors, labels in loader:
            images, vectors, labels = images.to(device), vectors.float().to(device), labels.to(device)
            outputs = model(images, vectors)

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
    print(f"Device: {device} | Fusion Method: {args.fusion_method.upper()}")

    if not os.path.exists(args.csv_file):
        raise FileNotFoundError("Error: CSV file not found.")
        
    df = pd.read_csv(args.csv_file)
    le_class = LabelEncoder()
    df['class_encoded'] = le_class.fit_transform(df['class'])
    num_classes = len(le_class.classes_)

    embedding_map, emb_dim = load_embedding_map(args.embedding_path, key_col='h3_index')

    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['class'])

    print(f"Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    t_train = transforms.Compose([
        transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t_eval = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = EuroSATDataset(train_df, args.data_root, embedding_map, emb_dim, t_train, le_class)
    val_set = EuroSATDataset(val_df, args.data_root, embedding_map, emb_dim, t_eval, le_class)
    test_set = EuroSATDataset(test_df, args.data_root, embedding_map, emb_dim, t_eval, le_class)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultimodalEuroSATModel(num_classes=num_classes, input_embedding_dim=emb_dim, fusion_method=args.fusion_method).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\nStarting Training...")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {t_acc:.2%} Loss: {t_loss:.4f} | Val Acc: {v_acc:.2%} Loss: {v_loss:.4f}")
        best_val_acc = max(best_val_acc, v_acc)

    print("\n" + "="*30 + "\nFINAL TEST REPORT\n" + "="*30)
    test_loss, test_acc, true_labels, pred_labels = evaluate(model, test_loader, criterion, device)
    report = classification_report(true_labels, pred_labels, target_names=le_class.classes_, digits=4)
    print(f"Test Accuracy: {test_acc:.2%}\nTest Loss:     {test_loss:.4f}\n\nDetailed Classification Report:\n{report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="eurosat_h3_index.csv", help="Main CSV with filenames and H3 indexes")
    parser.add_argument("--data_root", type=str, default="EuroSAT_MS", help="Image folder")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to .pkl/.csv with embeddings")
    parser.add_argument("--fusion_method", type=str, choices=['concat', 'attention', 'transformer', 'bilinear'], default='transformer', help="Method to fuse modalities")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    run_experiment(args)