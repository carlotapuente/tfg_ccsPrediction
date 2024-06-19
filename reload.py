
import pandas as pd
import pickle

with open("features_dict.pkl", "rb") as f:
    features_dict = pickle.load(f)


features_dict.keys()
features_dict["id_cols"] = features_dict["id_cols"]["inchi"].values

with open("features_dict_v2.pkl", "wb") as f:
    pickle.dump(features_dict, f)