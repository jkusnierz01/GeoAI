# models.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(SimpleMLP, self).__init__()
        # 1. Smart Projection Layer (Compresses input -> hidden_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3) # Prevents overfitting on high-dim embeddings
        
        # 2. Processing Layers
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.relu = nn.ReLU()
        
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

        # 3. Output Head
        self.output = nn.Linear(128, 1)  # Single output for regression
    def forward(self, x):
        # Apply projection
        x = self.projection(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Deep processing
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        
        return self.output(x)

class DeepRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=50, lr=0.001, batch_size=64, device=None):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        # 1. Scale Data (CRITICAL for mixing embeddings with counts)
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to Tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        # 2. Initialize Model
        input_dim = X.shape[1]
        self.model = SimpleMLP(input_dim).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        # 3. Training Loop
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Training MLP on {self.device} for {self.epochs} epochs...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            with tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False) as t:
                for batch_X, batch_y in t:
                    optimizer.zero_grad()
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    t.set_postfix(loss=loss.item())
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch+1}/{self.epochs} - Avg Loss: {total_loss / len(loader):.4f}")
                
        return self

    def predict(self, X):
        self.model.eval()
        # Scale inputs using the scaler fitted on Train
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
            
        return preds