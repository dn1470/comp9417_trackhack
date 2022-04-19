import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import model_selection, ensemble
from hyperparameter_tuning import param_tuning
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

'''Hyperparameter tuning'''
def xgb_hyperTun():
    data = pd.read_csv("extracted_features/extracted_features-kfold-final.csv")
    feature = data['ebb_eligible']

    # Remove categorical features
    data = data.drop(columns= ['ebb_eligible', 'customer_id'])

    params_grid = {
        'seed': [0],
        'alpha': [0, 5, 10],
        'learning_rate': [0.1, 0.25, 0.5, 0.75, 1.0],
        'colsample_bytree': [0.25, 0.75, 1],
        'max_depth': [1, 5, 10],
        'verbosity': [0],
        'use_label_encoder': [False],
        'eval_metric': ['logloss']
    }

    param_tuning(xgb.XGBClassifier, params_grid, data, feature)

# Hyperparameter tuning
xgb_hyperTun()

# Create a training set & Test set
ebb1_df = pd.read_csv('./extracted_features-ebb_set1.csv')
ebb2_df = pd.read_csv('./extracted_features-ebb_set2.csv')
eval_df = pd.read_csv('./extracted_features-eval_set.csv')
all_df = pd.concat([ebb1_df, ebb2_df])

columns_to_drop = ['last_redemption_date', 'first_activation_date', 'year', 'operating_system', 'state', 'latest_activation_channel',
                   'latest_activation_year', 'latest_activation_month', 'latest_activation_day', 'language']

# drop time and categorical features
all_df = all_df.drop(columns=columns_to_drop)

eval_df = eval_df.drop(columns=columns_to_drop)

# fill null values with 0 (for ebb_eligible)
# Treat unlabelled data as negatives
all_df = all_df.fillna(0)

X_train = all_df.drop(columns=['ebb_eligible', 'customer_id']).values
y_train = all_df['ebb_eligible'].values

X_eval = eval_df.drop(columns=['customer_id']).values

# XGBoost classifier
xg_reg = xgb.XGBClassifier(colsample_bytree = 0.3, learning_rate = 0.2, max_depth = 20, alpha = 10, subsample=0.5, n_estimators = 10, use_label_encoder=False, eval_metric='logloss')

# Fit the training set
xg_reg.fit(X_train,y_train)

# Predict the test set and save the result in csv to submit
y_pred = xg_reg.predict(X_eval)
predictions = np.array([round(value) for value in y_pred])
result = pd.DataFrame({'customer_id': eval_df['customer_id'], 'ebb_eligible': predictions})
result.to_csv('../submission/2022-04-15_x1.csv', index=False)