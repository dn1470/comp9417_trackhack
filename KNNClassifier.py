from sklearn import neighbors
from sklearn import model_selection
from sklearn import metrics
import pandas as pd
from datetime import date

def kNN(k):
    ''' Creates a KNN algorithm that fits to the ebb set with neighbours set to a given k value, and saves the result to a csv.'''
    print("K Neighbours for k = {}".format(k))
    # Read the datasets and concatenate
    data = pd.read_csv("extracted_features-ebb_set1.csv")
    data = pd.concat([data, pd.read_csv("extracted_features-ebb_set2.csv")])
    # Fill missing values with zero
    feature = data['ebb_eligible'].fillna(0)
    # Drop unnecessary columns
    data = data.drop(columns= ['ebb_eligible', 'customer_id', 'last_redemption_date', 'first_activation_date', 'latest_activation_channel', 'language', 'state'])
    # minkowski metric nearest neighbours with given k as the number of neighbours
    nbrs = neighbors.KNeighborsClassifier(n_neighbors = k)

    nbrs.fit(data, feature)

    eval_data = pd.read_csv("extracted_features-eval_set.csv")
    eval_data = eval_data.set_index("customer_id")

    eval_f = eval_data.drop(columns= ['last_redemption_date', 'first_activation_date', 'latest_activation_channel', 'language', 'state'])
    # Predic the value as an int (should be 1 or 0)
    eval_data["ebb_eligible"] = nbrs.predict(eval_f).astype("int")

    result = eval_data[["ebb_eligible"]].copy()
    result.to_csv("../submission/" + str(date.today()) + "knn_{}.csv".format(k))
