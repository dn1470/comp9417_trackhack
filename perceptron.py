import pandas as pd
from sklearn import linear_model
from sklearn import model_selection

def perceptron():
    ''' Trains a perceptron with a random state set to 0, and prints the result to a csv'''
    data = pd.read_csv("extracted_features/extracted_features-ebb_set1-state_distribution.csv")
    data = pd.concat([data, pd.read_csv("extracted_features/extracted_features-ebb_set2-state_distribution.csv")])
    feature = data['ebb_eligible'].fillna(0)

    # Remove categorical features
    data = data.drop(columns= ['ebb_eligible', 'customer_id', 'last_redemption_date', 'first_activation_date', 'latest_activation_channel', 'language'])
    # Set with a consistent random state for repeated reliable submissions
    classifier = linear_model.Perceptron(random_state=0)
    # Fit the data
    classifier = classifier.fit(data, feature)

    eval_data = pd.read_csv("extracted_features/extracted_features-eval_set-state_distribution.csv")
    eval_data = eval_data.set_index("customer_id")

    eval_f = eval_data.drop(columns= ['last_redemption_date', 'first_activation_date', 'latest_activation_channel', 'language'])
    eval_data["ebb_eligible"] = classifier.predict(eval_f).astype("int")

    result = eval_data[["ebb_eligible"]].copy()

    result.to_csv("../submission/2022-04-12_p_1.csv")