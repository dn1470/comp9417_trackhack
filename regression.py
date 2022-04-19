import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def regression():
    extracted_ebb1 = pd.read_csv('./extracted_features-ebb_set1.csv')
    extracted_ebb2 = pd.read_csv('./extracted_features-ebb_set2.csv')
    extracted_ebb = pd.concat([extracted_ebb1, extracted_ebb2])

    dropped_columns = ['last_redemption_date', 'first_activation_date', 'year', 'operating_system', 'state', 'latest_activation_channel',
                    'latest_activation_year', 'latest_activation_month', 'latest_activation_day', 'language', 'customer_id']
    extracted_ebb = extracted_ebb.drop(columns=dropped_columns)
    extracted_ebb[extracted_ebb < 0] = 0
    # reindex the ebb to avoid duplicate indexes
    extracted_ebb = extracted_ebb.reset_index().drop(columns='index')

    target = 'ebb_eligible'
    labeled_ebb = extracted_ebb[~np.isnan(extracted_ebb[target])]
    unlabeled_ebb = extracted_ebb[np.isnan(extracted_ebb[target])]

    # Step 1: Identify 'reliable negative'(RN) examples

    # randomly select some positive examples (spies) and put them together with unlabeled datasets    
    # initially treat all the unlabeled data as negative
    # Training a naive bayes classifier and update using expectation maximization

    # reliable negative examples are the unlabled negative examples 
    # for which the posterior probability is lower than the posterior probability of any of the spies

    s = 0.7 # sample ratio

    spies_df = labeled_ebb.sample(frac=s, random_state=1)
    y_spies = spies_df[target]
    index_spies = spies_df.index

    Ps_df = pd.concat([labeled_ebb, spies_df]).drop_duplicates(keep=False)
    y_Ps = Ps_df[target]

    unlabeled_ebb = unlabeled_ebb.assign(ebb_eligible=0) # initially set label 0 for unlabeled data
    index_unlabeled = unlabeled_ebb.index

    Us_df = pd.concat([unlabeled_ebb, spies_df])
    y_Us = Us_df[target]

    Us_df = Us_df.drop(columns=target)
    spies_df = spies_df.drop(columns=target)
    Us_np = Us_df.to_numpy()
    spies_np = spies_df.to_numpy()

    nb = GaussianNB(var_smoothing=0.01)
    nb.fit(Us_np, y_Us)

    # get the lowest posterior prob for spies
    min_spies_prob = 1
    for i in index_spies:
        prob_s = nb.predict_proba(spies_df.loc[i].to_numpy().reshape(1, -1))
        if (prob_s[0,1] < min_spies_prob):
            min_spies_prob = prob_s[0,1]
    min_spies_prob

    # update labels to get RN
    for j in index_unlabeled:
        prob_u = nb.predict_proba(Us_df.loc[j].to_numpy().reshape(1, -1))
        if (prob_u[0,1] > (min_spies_prob+0.03)):
            unlabeled_ebb.at[j, target] = 1
            
    reliable_negative = unlabeled_ebb[unlabeled_ebb['ebb_eligible'] == 0]
            
    unique, counts = np.unique(unlabeled_ebb['ebb_eligible'], return_counts=True)
    result = np.column_stack((unique, counts)) 

    # Step 2: Fit Positive and RN data to logistic regression model

    # train the logistic regression model on P and RN
    new_set = pd.concat([Ps_df, reliable_negative])
    y_new_set = new_set[target]
    new_set_df = new_set.drop(columns=target)

    clf = LogisticRegression(max_iter=10000, C=0.01)
    clf.fit(new_set_df.values, y_new_set)

    # test on eval_set
    eval_df = pd.read_csv('extracted_features-eval_set.csv').drop(columns=['last_redemption_date', 'first_activation_date', 'year', 'operating_system', 'state', 'latest_activation_channel',
                    'latest_activation_year', 'latest_activation_month', 'latest_activation_day', 'language'])

    eval_test = eval_df.drop(columns=['customer_id']).values
    y_pred = clf.predict(eval_test).astype('int')
    result = pd.DataFrame({'customer_id': eval_df['customer_id'], 'ebb_eligible': y_pred})
    result.to_csv('../submission/2022-04-13_lr.csv', index=False)





