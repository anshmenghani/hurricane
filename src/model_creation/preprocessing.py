import pandas as pd
import numpy as np
import math

path = "/src/data/storms_added.csv"

def shuffle_data():
    df = pd.read_csv(path)
    batches = [group for _, group in df.groupby('name', sort=False)]
    np.random.shuffle(batches)
    shuffled_df = pd.concat(batches).reset_index(drop=True)
    shuffled_df.to_csv("storms_shuffle.csv")


def split_data(training_size, x_cols, y_cols):
    df = pd.read_csv("/Users/anshmenghani/Documents/GitHub/hurricane/src/model_creation/storms_shuffle.csv")
    num_rows = math.floor(len(df) * training_size)
    flag = True
    while flag: 
        if df.at[num_rows-1, "name"] == df.at[num_rows, "name"]:
            num_rows += 1
            continue
        else:
            flag = False
    test_df = df[:num_rows]
    train_df = df[num_rows:]
    x_train, y_train = train_df[x_cols], train_df[y_cols]
    x_test, y_test = test_df[x_cols], test_df[y_cols]
    x_train.to_csv("x_train.csv", index=False)
    x_test.to_csv("x_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

if __name__ == "__main__": 
    # shuffle_data()
    # split_data(0.2, ["year", "month", "day", "hour", "wind", "pressure", "status_disturbance", "status_extratropical", "status_hurricane", "status_other low", "status_subtropical depression", "status_subtropical storm", "status_tropical depression", "status_tropical storm", "status_tropical wave", "bearing_last_deg", "displacment_km", "speed_kmh", "last_loc_lat", "last_loc_long", "land"], ["lat", "long"])
    pass
