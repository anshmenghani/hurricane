import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# === 1. Define a simple regression model ===
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output shape: [batch_size, 2]
        )

    def forward(self, x):
        return self.model(x)

# === 2. Load input and output data ===
x_df = pd.read_csv("x_train.csv")
y_df = pd.read_csv("y_train.csv")

# Drop NaNs from x and match y by index
x_df_clean = x_df.dropna()
y_df_clean = y_df.loc[x_df_clean.index]

# Convert to NumPy
x_train = x_df_clean.values.astype("float32")
y_train = y_df_clean.values.astype("float32")



x_tensor = torch.tensor(x_train)
y_tensor = torch.tensor(y_train)

# === 3. Prepare dataset and loader ===
dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# === 4. Initialize model, loss, optimizer ===
model = SimpleRegressor(input_dim=x_tensor.shape[1])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# === 5. Train loop ===
for epoch in range(100):
    model.train()
    epoch_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)  # preds shape: [batch_size, 2]
        loss = loss_fn(preds, yb)  # yb shape: [batch_size, 2]
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(loader):.4f}")

# === 6. Save model ===
torch.save(model.state_dict(), "simple_model.pth")
