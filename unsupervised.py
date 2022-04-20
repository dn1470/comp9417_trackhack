import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt

# metrics to evaluate cluistering models
def stats(y_true, y_pred):
    agreed_pairs = 0
    for y_i, y_pred_i in zip(y_true, y_pred):
        if y_i == 1 and y_pred_i == 1:
            agreed_pairs += 1
    agreed_ratio = agreed_pairs / np.sum(y_true)
    positive_rate = np.sum(y_pred) / y_pred.shape[0]
    print(f"agreed pairs ratio: {100 * max(agreed_ratio, 1 - agreed_ratio)}%")
    print(f"0, 1 ratio in y_pred: {100 * max(positive_rate, 1 - positive_rate)}%")


pd.set_option("expand_frame_repr", False)

# choose a subset of features
columns_to_drop = ["language"]

# columns_to_drop = ['activation_count',
#  'year',
#  'latest_activation_channel',
#  'memory_total',
#  'storage_available',
#  'storage_used',
#  'bluetooth_on',
#  'operating_system',
#  'storage_total',
#  'has_leased',
#  'language']

# columns_to_drop = [ 'last_redemption_date', 'first_activation_date', 'year', 'state',
#                     'latest_activation_channel',
#                     'storage_total', 'mean_voice_minutes', 'mean_total_sms','memory_total',
#                     'is_auto_refill_enrolled', 'has_leased', 'bluetooth_on', 'language', 'storage_available', 'storage_used',
#                     'mean_total_kb', 'mean_hotspot_kb']

ebb1_df = pd.read_csv("extracted_features/extracted_features-ebb_set1-all.csv")
ebb2_df = pd.read_csv("extracted_features/extracted_features-ebb_set2-all.csv")
ebb_df = pd.concat([ebb1_df, ebb2_df])

ebb_df = ebb_df.drop(columns=columns_to_drop)
ebb_df = ebb_df.fillna(0)

ebb_df.shape


# get X_train and y_train as numpy arrays
y_train = ebb_df["ebb_eligible"].to_numpy()
X_train = ebb_df.drop(columns=["customer_id", "ebb_eligible"]).to_numpy()

# fit X_train to standard scaler and apply pca
X_train = StandardScaler().fit_transform(X_train)
pca = PCA(n_components=20)
X_train_reduced = pca.fit_transform(X_train)
pca.explained_variance_ratio_

# the total explained variance ration of using first i principal components
total_explained_variance_ratio = [
    np.sum(pca.explained_variance_ratio_[: i + 1])
    for i in range(len(pca.explained_variance_ratio_))
]

fig, ax1 = plt.subplots(figsize=(15, 10), dpi=80)
ax1.plot(np.arange(1, 21), pca.explained_variance_ratio_, "o-", linewidth=2)
ax1.set_xticks(list(range(21)))
ax1.set_xlabel("ith principle component", fontsize=14)
ax1.set_ylabel("explained variance ratio", fontsize=14)

fig, ax2 = plt.subplots(figsize=(15, 10), dpi=80)
ax2.plot(np.arange(1, 21), total_explained_variance_ratio, "o-", linewidth=2)
ax2.set_xticks(list(range(21)))
ax2.set_xlabel("n principle components", fontsize=14)
ax2.set_ylabel("total explained variance ratio", fontsize=14)

# fit to Gaussian Mixture model
gm = GaussianMixture(
    n_components=2, random_state=0, covariance_type="full", max_iter=200
)
gm.fit(X_train_reduced)
y_cluster = gm.predict(X_train_reduced)
stats(y_train, y_cluster)

# fit to k-means model
y_cluster = KMeans(n_clusters=2, random_state=0).fit_predict(X_train_reduced)
stats(y_train, y_cluster)

# OPTIONAL, if predicted labels are reverted, reverse them
y_cluster = 1 - y_cluster
stats(y_train, y_cluster)

# replace predictions for labeled samples with 1
for i, y_i in enumerate(y_train):
    if y_i == 1:
        y_cluster[i] = 1
stats(y_train, y_cluster)

# then train classification models on generated dataset
ebb_df["ebb_eligible"] = y_cluster
