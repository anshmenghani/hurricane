import torch
import torch.nn as nn
import pandas as pd

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
