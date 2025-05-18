import pandas as pd
import math
from geopy.distance import geodesic
from global_land_mask import globe

fn = "storms_sampled_oneHot"
df = pd.read_csv(fn + ".csv")

def drop_bad_cols(): # drop unusable data
    global df
    df = df.drop(["category", "tropicalstorm_force_diameter", "hurricane_force_diameter"], axis=1)
    df.to_csv(fn + "_sampled.csv")

def categorize(col): # one hot the categorization of storm
    global df
    df_encoded = pd.get_dummies(df, columns=[col])
    # convert only the new one-hot columns to int
    one_hot_cols = [c for c in df_encoded.columns if c.startswith(col + '_')]
    df_encoded[one_hot_cols] = df_encoded[one_hot_cols].astype(int)
    df_encoded.to_csv(fn + "_sampled_oneHot.csv")

def add_new_features():
    global df 
    df["bearing_last_deg"] = [0] * len(df)
    df["displacment_km"] = [0] * len(df)
    df["speed_kmh"] = [0] * len(df)
    df["last_loc_lat"] = [0] * len(df)
    df["last_loc_long"] = [0] * len(df)
    df["land"] = [0] * len(df)
    for i, row in df.iterrows():
        if i == 0:
            continue
        if df.at[i-1, "name"] == df.at[i, "name"]:
            distance = geodesic((df.at[i, "lat"], df.at[i, "long"]), (df.at[i-1, "lat"], df.at[i-1, "long"])).km
            df.at[i, "bearing_last_deg"] = 180/math.pi * math.atan((df.at[i, "lat"] - df.at[i-1, "lat"]) / (df.at[i, "long"] - df.at[i-1, "long"]))
            df.at[i, "displacment_km"] = distance
            df.at[i, "speed_kmh"] = distance / 6
            df.at[i, "last_loc_lat"] = df.at[i-1, "lat"]
            df.at[i, "last_loc_long"] = df.at[i-1, "long"]
        if globe.is_land(df.at[i, "lat"], df.at[i, "long"]):
            df.at[i, "land"] = 1

if __name__ == "__main__":
    # drop_bad_cols()
    # categorize()
    # add_new_features()
    # df.to_csv("storms_added.csv")
    pass
