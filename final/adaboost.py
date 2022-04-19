from sklearn import ensemble
from utils import predict

def adaboost(X_train, y_train):
    '''
    Final tuned model using AdaBoost
    '''
    adb_clf = ensemble.AdaBoostClassifier(random_state=0, n_estimators=125, learning_rate=1.25)
    adb_clf = adb_clf.fit(X_train, y_train)

    adb_pred = predict(adb_clf)
    adb_score = adb_clf.score(X_train, y_train)

    return adb_pred, adb_score
