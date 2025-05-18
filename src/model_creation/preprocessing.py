import pandas as pd
import numpy as np

path = "/src/data/storms_added.csv"

def shuffle_data():
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  #remove unnamed columns
    batches = [group for _, group in df.groupby('name', sort=False)]
    np.random.shuffle(batches)
    shuffled_df = pd.concat(batches).reset_index(drop=True)
    shuffled_df.to_csv("storms_shuffle.csv", index=False)

def create_k_folds(k=5):
    df = pd.read_csv("storms_shuffle.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  #remove unnamed columns
    hurricane_groups = [group for _, group in df.groupby("name")]
    np.random.shuffle(hurricane_groups)

    folds = [[] for _ in range(k)]
    for i, group in enumerate(hurricane_groups):
        folds[i % k].append(group)

    for i, fold in enumerate(folds):
        fold_df = pd.concat(fold).reset_index(drop=True)
        fold_df.to_csv(f"model/fold_{i+1}.csv", index=False)

if __name__ == "__main__":
    shuffle_data()
    create_k_folds(k=5)
