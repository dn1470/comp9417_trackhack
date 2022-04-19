import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

def self_train(all_training_data):
    '''
    K-fold self-training method used on the training dataset
    '''
    all_training_data = all_training_data.drop(columns=['language'])

    labeled_samples = all_training_data.loc[all_training_data['ebb_eligible'] == 1]
    unlabeled_samples = all_training_data.loc[all_training_data['ebb_eligible'] == 0]

    nfolds = 10
    ntargets = 5000
    label_preds = np.array(())
    unlabeled_train_size = unlabeled_samples.shape[0] - ntargets

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
        
    unlabeled_samples_pred = unlabeled_samples.copy()
    unlabeled_samples_pred['ebb_eligible'] = label_preds
    unlabeled_samples_pred = unlabeled_samples_pred.loc[unlabeled_samples_pred['ebb_eligible'] == 0]

    train_df_final = pd.concat([labeled_samples, unlabeled_samples_pred])

    return train_df_final
