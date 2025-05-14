import pandas as pd

# Load original CSVs
x_df = pd.read_csv("x_test.csv")
y_df = pd.read_csv("y_test.csv")

# Drop rows with NaNs in x_df
x_clean = x_df.dropna()

# Align y_df with x_clean (keep same indices)
y_clean = y_df.loc[x_clean.index]

# Save cleaned files
x_clean.to_csv("x_test_clean.csv", index=False)
y_clean.to_csv("y_train_clean.csv", index=False)
