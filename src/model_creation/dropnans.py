import pandas as pd

x_df = pd.read_csv("x_test.csv")
y_df = pd.read_csv("y_test.csv")

x_clean = x_df.dropna()

y_clean = y_df.loc[x_clean.index]

x_clean.to_csv("x_test_clean.csv", index=False)
y_clean.to_csv("y_train_clean.csv", index=False)
