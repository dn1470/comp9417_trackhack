import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from hyperparameter_tuning import param_tuning

'''Hyperparameter tuning'''
def hist_hyperTun():
    data = pd.read_csv("extracted_features/extracted_features-kfold-final.csv")
    feature = data['ebb_eligible']

    # Remove categorical features
    data = data.drop(columns= ['ebb_eligible', 'customer_id'])

    params_grid = {
        'random_state': [0],
        'min_samples_leaf': [5, 10, 15, 20],
        'learning_rate': [0.1, 0.25, 0.5, 0.75, 1.0],
        'max_depth': [1, 5, 10],
        'verbose': [False]
    }

    param_tuning(ensemble.HistGradientBoostingClassifier, params_grid, data, feature)

# Hyperparameter tuning
hist_hyperTun()

# Create a training set & Test set
data = pd.read_csv("extracted_features/extracted_features-ebb_set1-all.csv")
data = pd.concat([data, pd.read_csv("extracted_features/extracted_features-ebb_set2-all.csv")])

# fill null values with 0 (for ebb_eligible)
# Treat unlabelled data as negatives
feature = data['ebb_eligible'].fillna(0)

# Remove categorical features
data = data.drop(columns= ['ebb_eligible', 'customer_id', 'language'])

# HistGradientBoost classifier
classifier = ensemble.HistGradientBoostingClassifier(random_state=0, verbose=1, max_iter=120)

# Fit the training set
classifier = classifier.fit(data, feature)

# Predict the test set and save the result in csv to submit
eval_data = pd.read_csv("extracted_features/extracted_features-eval_set-all.csv")
eval_data = eval_data.set_index("customer_id")

eval_f = eval_data.drop(columns= ['language'])
eval_data["ebb_eligible"] = classifier.predict(eval_f).astype("int")

result = eval_data[["ebb_eligible"]].copy()

result.to_csv("../submission/2022-04-13_d_1.csv")