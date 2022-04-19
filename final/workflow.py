import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import date
from feature_extraction import feature_extraction
from self_train import self_train
from feature_selection import feature_selection
from utils import train_setup
from decisiontree import decisiontree
from randomforest import randomforest
from xgboost import xgboost
from catboostclassifier import catboost
from adaboost import adaboost
from histgradientboost import histgradientboost
from ensemblemethod import ensemblemethod

def extract_features():
    feature_extraction = feature_extraction()
    feature_extraction = feature_extraction()
    ebb_set1 = feature_extraction.merge_dataset("ebb_set1")
    ebb_set2 = feature_extraction.merge_dataset("ebb_set2")

    eval_set = feature_extraction.merge_dataset("eval_set").set_index("customer_id")
    eval_set = eval_set.drop(columns= ['language'])

    all_training_data = pd.concat([ebb_set1, ebb_set2])
    all_training_data = all_training_data.fillna(0)

    return all_training_data, eval_set


if __name__ == "__main__":
    all_training_data, eval_set = extract_features()
    train_df_final = self_train(all_training_data)
    train_df_final, eval_set = feature_selection(train_df_final, eval_set)
    X_train, y_train = train_setup(train_df_final)

    # Individual models
    dt_pred, dt_score = decisiontree(X_train, y_train)
    rf_pred, rf_score = randomforest(X_train, y_train)
    xgb_pred, xgb_score = xgboost(X_train, y_train)
    cb_pred, cb_score = catboost(X_train, y_train)
    adb_pred, adb_score = adaboost(X_train, y_train)
    hgb_pred, hgb_score = histgradientboost(X_train, y_train)

    ensemblemethod(dt_pred, rf_pred, xgb_pred, cb_pred, adb_pred, hgb_pred)









    


