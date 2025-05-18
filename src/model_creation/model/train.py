import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

#dropout layer
class GaussianDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.std = (p / (1 - p)) ** 0.5 #Gaussian deviation
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        noise = torch.randn_like(x) * self.std + 1
        return x * noise

#feedforward neural network
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressor, self).__init__()
        alpha = 0.1 #rate of dropout
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64) #ReLU activation
            nn.PReLU(),
            nn.LayerNorm(64), #layer normalization
            GaussianDropout(alpha),

            nn.Linear(64, 32),
            nn.PReLU(),
            nn.LayerNorm(32),
            GaussianDropout(alpha),

            nn.Linear(32, 32),
            nn.PReLU(),
            nn.LayerNorm(32),
            GaussianDropout(alpha),

            nn.Linear(32, 32),
            nn.PReLU(),
            nn.LayerNorm(32),
            GaussianDropout(alpha),

            nn.Linear(32, 2) #two-coordinate output
        )

    def forward(self, x):
        return self.model(x)

#data from folds
def load_folds(folds):
    df = pd.concat([pd.read_csv(f"fold_{i}.csv") for i in folds])
    df = df.dropna()
    x = df[["year", "month", "day", "hour", "wind", "pressure", "status_disturbance", "status_extratropical", "status_hurricane", "status_other low", "status_subtropical depression", "status_subtropical storm", "status_tropical depression", "status_tropical storm", "status_tropical wave", "bearing_last_deg", "displacment_km", "speed_kmh", "last_loc_lat", "last_loc_long", "land"]].values.astype("float32")
    y = df[["lat", "long"]].values.astype("float32")
    return x, y

#main training loop
def main(training_folds, testing_fold):
    #load training and testing data
    x_train, y_train = load_folds(training_folds)
    x_test, y_test = load_folds(testing_fold)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_tensor = torch.tensor(x_test)

    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SimpleRegressor(input_dim=x_train.shape[1])
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    #training loop
    for epoch in range(100000):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad() #clear gradient
            preds = model(xb) #pass foward
            loss = loss_func(preds, yb) #find loss
            loss.backward() #push loss back
            optimizer.step() #update weights
            epoch_loss += loss.item() #add loss
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    # torch.save(model.state_dict(), "simple_model.pth")

    model.eval()
    with torch.no_grad():
        preds = model(test_tensor).numpy()

    pred_df = pd.DataFrame(preds, columns=["pred_lat", "pred_long"])
    pred_df.to_csv(f"predictions_fold{testing_fold[0]}.csv", index=False)


if __name__ == "__main__":
    main([1, 3, 4, 5], [2])
    # folds_l = list(range(1, 6))
    # for i in folds_l:
    #     folds_l.remove(i)
    #     main(folds_l, [i])
    #     folds_l = list(range(1, 6))
