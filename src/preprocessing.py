import pandas as pd

fn = "storms"
df = pd.read_csv(fn + ".csv")

def drop_bad_cols():
    global df
    df = df.drop(["category", "tropicalstorm_force_diameter", "hurricane_force_diameter"], axis=1)
    df.to_csv(fn + "_sampled.csv")

def categorize(col):
    global df
    df_encoded = pd.get_dummies(df, columns=[col])
    # Convert only the new one-hot columns to int
    one_hot_cols = [c for c in df_encoded.columns if c.startswith(col + '_')]
    df_encoded[one_hot_cols] = df_encoded[one_hot_cols].astype(int)
    df_encoded.to_csv(fn + "_sampled_oneHot.csv")

if __name__ == "__main__":
    drop_bad_cols()
    categorize("status")
