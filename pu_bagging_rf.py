import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt        # for plotting
%matplotlib inline  
import warnings
warnings.simplefilter(action='error', category=FutureWarning)
from pulearn import BaggingPuClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Training Set
data = pd.read_csv("extracted_features-ebb_set1.csv")
data = pd.concat([data, pd.read_csv("extracted_features-ebb_set2.csv")])

# fill null values with 0 (for ebb_eligible)
# Treat unlabelled data as negatives
feature = data['ebb_eligible'].fillna(0)

# Test Set
eval_data = pd.read_csv("extracted_features-eval_set.csv")
eval_data = eval_data.set_index("customer_id")

# Remove customer id and class
data = data.drop(columns= ['ebb_eligible', 'customer_id'])

# Convert last_redemption_date, first_activation_date

# Convert date columns to timestamps
def convert_col_to_EPOCH(col):
    dates = pd.to_datetime(col, format='%Y-%M-%d')
    return (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

# Training Set
data['last_redemption_date'] = convert_col_to_EPOCH(data['last_redemption_date'])
data['first_activation_date'] = convert_col_to_EPOCH(data['first_activation_date'])

# Test Set
eval_data['last_redemption_date'] = convert_col_to_EPOCH(eval_data['last_redemption_date'])
eval_data['first_activation_date'] = convert_col_to_EPOCH(eval_data['first_activation_date'])

# State

# Training Set
# Fill the missing value with the most frequent value
st = data["state"].value_counts().idxmax()
data["state"] = data["state"].replace(np.nan, st)

# Label Encoding
label = LabelEncoder()
data["state"] = label.fit_transform(data["state"])

# Test Set
# Fill the missing value with the most frequent value
st2 = eval_data["state"].value_counts().idxmax()
eval_data["state"] = eval_data["state"].replace(np.nan, st2)

# Label Encoding
eval_data["state"] = label.fit_transform(eval_data["state"])

# latest_activation_channel

label = LabelEncoder()
# Training Set
#Fill the missing value with the most frequent value
channel = data["latest_activation_channel"].value_counts().idxmax()
data["latest_activation_channel"] = data["latest_activation_channel"].replace(np.nan, channel)


# Label Encoding
data["latest_activation_channel"] = label.fit_transform(data["latest_activation_channel"])

# Test Set
#Fill the missing value with the most frequent value
channel_eval = eval_data["latest_activation_channel"].value_counts().idxmax()
eval_data["latest_activation_channel"] = eval_data["latest_activation_channel"].replace(np.nan, channel_eval)


# Label Encoding
eval_data["latest_activation_channel"] = label.fit_transform(eval_data["latest_activation_channel"])

# language - Drop
# Total Missing values in data["language"] = 0.6384
remove = data.drop("language",axis=1, inplace=True)

# Check the contents of the set
print('%d data points and %d features' % (data.shape))
print('%d positive out of %d total' % (sum(feature), len(feature)))

'''
Tune max_smples

Hide some positive values from Training set to check how well the model indentify the positives
Test the accuracy
'''
# Hide some positive values to check how well the model identify the positives

y_orig = feature.copy()
y = feature.copy()

hidden_size = round(sum(feature)*0.2)

y.loc[
    np.random.choice(
        y[y == 1].index, 
        replace = False, 
        size = hidden_size
    )
] = 0

# Bagging with RandomForest
rf = BaggingPuClassifier(RandomForestClassifier(random_state=0, max_depth=20), 
                         # n_estimators = 50, 
                         n_jobs = -1, 
                         max_samples = 24000
                        )

# Fit training set 
rf.fit(data,y)

# Check the new contents of the set
print('%d positive out of %d total' % (sum(y), len(y)))

print('Precision: ', precision_score(y_orig, rf.predict(data)))
print('Recall: ', recall_score(y_orig, rf.predict(data)))
print('Accuracy: ', accuracy_score(y_orig, rf.predict(data)))

'''Fit the model on the original training set to predict the class of test set'''
eval_f = eval_data.drop(columns= ['language'])
classifier = rf.fit(data, feature)


eval_data["ebb_eligible"] = classifier.predict(eval_f).astype("int")

result = eval_data[["ebb_eligible"]].copy()

result.to_csv("../submission/2022-04-12_y_1.csv")
