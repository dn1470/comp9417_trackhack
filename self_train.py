import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
%matplotlib inline

# Create a training set & Test set
ebb1_df = pd.read_csv('extracted_features/extracted_features-ebb_set1-all.csv')
ebb2_df = pd.read_csv('extracted_features/extracted_features-ebb_set2-all.csv')
all_df = pd.concat([ebb1_df, ebb2_df])

# fill null values with 0 (for ebb_eligible)
# Treat unlabelled data as negatives
all_df = all_df.fillna(0)

pd.set_option('expand_frame_repr', False)

all_df = all_df.drop(columns=['language'])

labeled_samples = all_df.loc[all_df['ebb_eligible'] == 1]
unlabeled_samples = all_df.loc[all_df['ebb_eligible'] == 0]

print(f'unlabeled_samples.shape: {unlabeled_samples.shape}')

nfolds = 10
ntargets = 5000
label_preds = np.array(())
unlabeled_train_size = unlabeled_samples.shape[0] - ntargets

# CatBoost with K-fold
for curr_iter in range(nfolds):
    start, end = curr_iter * ntargets, (curr_iter + 1) * ntargets
    train_df = pd.concat([labeled_samples.sample(n=unlabeled_train_size), unlabeled_samples.iloc[:start], unlabeled_samples.iloc[end:]])
    pred_df = unlabeled_samples.iloc[start:end]
    
    X_train = train_df.drop(columns=['customer_id', 'ebb_eligible']).values
    y_train = train_df['ebb_eligible'].values
    X_pred = pred_df.drop(columns=['customer_id', 'ebb_eligible']).values
    
    clf = CatBoostClassifier(logging_level='Silent')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_pred).astype('int')
    
    label_preds = np.hstack((label_preds, y_pred))
    
    print(f'y_pred.shape: {y_pred.shape}')
    print(f'label_preds.shape: {label_preds.shape}')
    
# Predict the test set and save the result in csv to submit
unlabeled_samples_pred = unlabeled_samples.copy()
unlabeled_samples_pred['ebb_eligible'] = label_preds
unlabeled_samples_pred = unlabeled_samples_pred.loc[unlabeled_samples_pred['ebb_eligible'] == 0]

train_df_final = pd.concat([labeled_samples, unlabeled_samples_pred])
train_df_final.to_csv('extracted_features/extracted_features-kfold-final.csv')

final_df = pd.read_csv('extracted_features/extracted_features-kfold-final.csv')