from sklearn import tree
from utils import predict

def decisiontree(X_train, y_train):
    '''
    Final tuned decision tree model
    '''
    dt_clf = tree.DecisionTreeClassifier(random_state=0, min_samples_leaf=19)
    dt_clf = dt_clf.fit(X_train, y_train)

    dt_pred = predict(dt_clf)
    dt_score = dt_clf.score(X_train, y_train)

    return dt_pred, dt_score
