import torch
import torch.nn as nn
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# === 1. Define the model (must match training definition) ===
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Final output shape: [batch_size, 2]
        )

    def forward(self, x):
        return self.model(x)

# === 2. Load input data ===
x_test = pd.read_csv("x_test.csv").dropna().values.astype("float32")  # or use "x_test.csv"
x_tensor = torch.tensor(x_test, dtype=torch.float32)

# === 3. Initialize and load the model ===
input_dim = x_tensor.shape[1]
model = SimpleRegressor(input_dim=input_dim)
model.load_state_dict(torch.load("simple_model.pth"))
model.eval()

# === 4. Run inference ===
with torch.no_grad():
    predictions = model(x_tensor)  # shape: [N, 2]

# === 5. Output results ===
pred_np = predictions.numpy()
print(pred_np)

# Optional: save to CSV
pd.DataFrame(pred_np, columns=["pred_lat", "pred_lon"]).to_csv("predictions.csv", index=False)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold = 1
mse_list = []

for train_idx, val_idx in kf.split(x_tensor):
    print(f"\n=== Fold {fold} ===")
    
    x_train, x_val = x_tensor[train_idx], x_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
    
    input_dim = x_train.shape[1]
    model = SimpleRegressor(input_dim=input_dim)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        preds = model(x_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_preds = model(x_val)
        mse = mean_squared_error(y_val.numpy(), val_preds.numpy())
        print(f"Fold {fold} MSE: {mse:.4f}")
        mse_list.append(mse)
        fold += 1

print(f"\nAverage MSE across {k} folds: {np.mean(mse_list):.4f}")

import matplotlib.pyplot as plt

# After your k-fold loop
folds = list(range(1, k + 1))

plt.figure(figsize=(8, 5))
plt.plot(folds, mse_list, marker='o', linestyle='-')
plt.title("K-Fold Mean Squared Error")
plt.xlabel("Fold")
plt.ylabel("MSE")
plt.grid(True)
plt.xticks(folds)
plt.show()

