import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

# train random forest models with different params

# small trees
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_eval).astype('int')
result = pd.DataFrame({'customer_id': eval_df['customer_id'], 'ebb_eligible': y_pred})
result.to_csv('../submission/2022-04-10_rf1.csv', index=False)

# larger trees
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_eval).astype('int')
result = pd.DataFrame({'customer_id': eval_df['customer_id'], 'ebb_eligible': y_pred})
result.to_csv('../submission/2022-04-10_rf2.csv', index=False)

clf = RandomForestClassifier(max_depth=10, min_samples_split=10, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_eval).astype('int')
result = pd.DataFrame({'customer_id': eval_df['customer_id'], 'ebb_eligible': y_pred})
result.to_csv('../submission/2022-04-10_rf3.csv', index=False)


clf = RandomForestClassifier(max_depth=20, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_eval).astype('int')
result = pd.DataFrame({'customer_id': eval_df['customer_id'], 'ebb_eligible': y_pred})
result.to_csv('../submission/2022-04-10_rf4.csv', index=False)

# no restrain
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_eval).astype('int')
result = pd.DataFrame({'customer_id': eval_df['customer_id'], 'ebb_eligible': y_pred})
result.to_csv('../submission/2022-04-10_rf5.csv', index=False)

#  train random forest models with different params
ebb1_df = pd.read_csv('extracted_features/extracted_features-ebb_set1-all.csv')
ebb2_df = pd.read_csv('extracted_features/extracted_features-ebb_set2-all.csv')
eval_df = pd.read_csv('extracted_features/extracted_features-eval_set-all.csv')
all_df = pd.concat([ebb1_df, ebb2_df])

columns_to_drop = ['language']

# drop time and categorical features
all_df = all_df.drop(columns=columns_to_drop)

eval_df = eval_df.drop(columns=columns_to_drop)

# fill null values with 0 (for ebb_eligible)
all_df = all_df.fillna(0)

X_train = all_df.drop(columns=['ebb_eligible', 'customer_id']).values
y_train = all_df['ebb_eligible'].values

X_eval = eval_df.drop(columns=['customer_id']).values

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_eval).astype('int')
result = pd.DataFrame({'customer_id': eval_df['customer_id'], 'ebb_eligible': y_pred})
result.to_csv('../submission/2022-04-13_rf_1.csv', index=False)