import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn import tree, ensemble
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import xgboost as xgb
from datetime import date
from feature_extraction import feature_extraction
from self_train import self_train
from feature_selection import feature_selection

'''
Utility functions
'''

def train_setup(train_df_final):
    '''
    Splits the train dataset into the feature and target variables
    '''
    y_train = train_df_final['ebb_eligible']
    X_train = train_df_final.drop(columns= ['ebb_eligible', 'customer_id'])

    return X_train, y_train

def predict(model, eval_set):
    '''
    Calls the given model's predict function
    Returns the model's predictions
    '''
    eval_copy = eval_set.copy()
    eval_copy['ebb_eligible'] = model.predict(eval_copy).astype('int')
    
    return eval_copy[['ebb_eligible']].copy()
