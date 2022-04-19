import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
%matplotlib inline

# Creating training set & test set
ebb1_df = pd.read_csv("extracted_features/extracted_features-ebb_set1-all.csv")
ebb2_df = pd.read_csv("extracted_features/extracted_features-ebb_set2-all.csv")
all_df = pd.concat([ebb1_df, ebb2_df])

# fill null values with 0 (for ebb_eligible)
# Treat unlabelled data as negatives
all_df = all_df.fillna(0)

columns_to_drop = ['language']

all_df = all_df.drop(columns=columns_to_drop)

X = all_df.drop(columns=['ebb_eligible', 'customer_id'])
y = all_df['ebb_eligible'].values

# Fit the training set
clf = CatBoostClassifier(logging_level='Silent')
clf.fit(X, y)

# To plot feature importances
features = list(zip(clf.feature_importances_, clf.feature_names_))
features.sort(key=lambda x: x[0], reverse=True)
fig, ax = plt.subplots(figsize=(30, 10))
ax.bar([f[1] for f in features], [f[0] for f in features])
ax.tick_params(labelrotation=90)

plt.show()