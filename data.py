#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from data_openml import data_split, DataSetCatCon

def load_data(n_fpts=None, n_rows=None, include_dimers=False):
    with open("features_dict_v2.pkl", "rb") as f:
        features_dict = pickle.load(f)

    y = features_dict["ccs"]

    X = features_dict["fingerprint"]
    if n_fpts is not None:
        X = X[:, :n_fpts]
    X = pd.DataFrame(X)
    X.columns = [f"fgp_{i}" for i in range(1, X.shape[1] + 1)]
    X.insert(0, "mz", features_dict["mz"])
    X.insert(1, "adduct", features_dict["adduct"])

    if not include_dimers:
        y = y[~X["adduct"].str.startswith("Dimer")]
        X = X[~X["adduct"].str.startswith("Dimer")]

    # Substitute adduct by ordinal encoder
    adduct_encoder = OrdinalEncoder()
    X["adduct"] = (
        adduct_encoder.fit_transform(X["adduct"].values.reshape(-1, 1)).astype("int")
    )

    n_adducts = len(adduct_encoder.categories_[0])
    if n_rows is not None:
        X = X.iloc[:n_rows]
        y = y[:n_rows]

    X = X.reset_index(drop=True)

    assert X.shape[0] == y.shape[0]

    return X, y


def data_prep(X, y, datasplit=[.65, .15, .2]):
    categorical_indicator = [False] + [True] * (X.shape[1] - 1)
    categorical_columns = X.columns[1:].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_dims = [3] + [2] * (X.shape[1] - 2) # [3] = categs adducts, [2] = categs fgpts
    cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.
 
    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    
    for col in categorical_columns:
        X[col] = X[col].astype("category")

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index
    
    y = pd.DataFrame(y).values

    X = X.drop(columns=['Set'])

    nan_mask = X.isna().astype(int)

    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)
    
    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    
    continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, continuous_mean_std

