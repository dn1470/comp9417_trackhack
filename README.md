# COMP9417 - TrackHack

## Team 09

| Team Member  | zId      |
| ------------ | -------- |
| Zhijie Zhu   | z5257738 |
| Zeyu Hou     | z5190728 |
| Finn Russell | z5259047 |
| Youngju Chae | z5339489 |
| Kaman Chan   | z5257026 |

---

### Final files

Please find the code used for the final model in the `/final` directory.

| File Name             | Description                                                                   |
| --------------------- | ----------------------------------------------------------------------------- |
| adaboost.py           | Final tuned model using AdaBoost                                              |
| catboostclassifier.py | Final tuned model using CatBoost                                              |
| decisiontree.py       | Final tuned decision tree model                                               |
| ensemblemethod.py     | Custom ensemble method which combines the predictions of multiple models      |
| feature_extraction.py | Condenses long form tables and merges all data together with the main dataset |
| feature_selection.py  | Feature selector - drops the unimportant features                             |
| histgradientboost.py  | Final tuned HistGradientBoostingClassifier                                    |
| randomforest.py       | Final random forest model                                                     |
| self_train.py         | K-fold self-training method used on the training dataset                      |
| utils.py              | Utility functions                                                             |
| workflow.py           | Main workflow - trigger the entire ensemble model from here                   |
| xgboost.py            | Final tuned model using XGBoostClassifier                                     |

---

### Other working files

| File Name                | Description                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| hyperparameter_tuning.py | Custom grid search function for hyperparameter tuning                                         |
| xgboost.py               | XGBoost model to fit training data set & predict the test set/hyperparameter tuning           |
| catboost.py              | CatBoost classifier with feature importance                                                   |
| unsupervised.py          | Unsupervised learning algorithms tested, including PCA, k-means and Gaussian Mixture          |
| self_train.py            | K-Fold self-training model which has a CatBoost base                                          |
| regression.py            | Regression model                                                                              |
| randomforest.py          | RandomForest model with different parameters                                                  |
| pu_bagging_rf.py         | PU Bagging with Random Forest/Parameter Tuning                                                |
| KNNClassifier.py         | KNN model                                                                                     |
| perceptron.py            | perceptron model                                                                              |
| feature_extraction.py    | Condenses long form tables and merges all data together with the main dataset                 |
| pu_bagging_dt.py         | PU Bagging with Decision Tree/Parameter Tuning                                                |
| histgradientboost.py     | HistGradientBoost model to fit training data set & predict the test set/hyperparameter tuning |
| boost_param.py           | AdaBoost feature importance & hyperparameter tuning/CatBoost hyperparameter tuning            |
