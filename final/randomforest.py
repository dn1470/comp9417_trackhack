from sklearn import ensemble
from utils import predict

def randomforest(X_train, y_train):
    rf_clf = ensemble.RandomForestClassifier(random_state=0)
    rf_clf = rf_clf.fit(X_train, y_train)

    rf_pred = predict(rf_clf)
    rf_score = rf_clf.score(X_train, y_train)

    return rf_pred, rf_score