from catboost import CatBoostClassifier
from utils import predict

def catboost(X_train, y_train):
    cb_clf = CatBoostClassifier(random_seed=0, n_estimators=100, learning_rate=1.0, max_depth=10, verbose=False)
    cb_clf = cb_clf.fit(X_train, y_train)

    cb_pred = predict(cb_clf)
    cb_score = cb_clf.score(X_train, y_train)

    return cb_pred, cb_score