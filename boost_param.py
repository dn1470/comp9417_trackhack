import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from hyperparameter_tuning import param_tuning
from catboost import CatBoostClassifier
from sklearn import model_selection
from itertools import product
import matplotlib.pyplot as plt
%matplotlib inline

# Create a training set to tune hyperparameter
data = pd.read_csv("extracted_features/extracted_features-kfold-final.csv")
feature = data['ebb_eligible']

# Remove categorical features
data = data.drop(columns= ['ebb_eligible', 'customer_id'])

# AdaBoost Hyperparameter Tuning
ada_params_grid = {
    'random_state': [0],
    'n_estimators': [50, 75, 100, 125],
    'learning_rate': [0.1, 0.25, 0.5, 0.75, 1.0, 1.25]
}

param_tuning(ensemble.AdaBoostClassifier, ada_params_grid, data, feature)

# Plot feature importance of AdaBoost
clf = ensemble.AdaBoostClassifier(random_state=0, n_estimators=125, learning_rate=1.25)
clf = clf.fit(data, feature)

features = list(zip(clf.feature_importances_, list(data.columns.values)))
features.sort(key=lambda x: x[0], reverse=True)
fig, ax = plt.subplots(figsize=(30, 10))
ax.bar([f[1] for f in features], [f[0] for f in features])
ax.tick_params(labelrotation=90)

# CatBoost Hyperparameter Tuning
params_grid = {
    'random_seed': [0],
    'n_estimators': [50, 75, 100, 125],
    'learning_rate': [0.1, 0.25, 0.5, 0.75, 1.0],
    'max_depth': [1, 5, 10],
    'verbose': [False]
}

param_tuning(CatBoostClassifier, params_grid, data, feature)