from sklearn import ensemble
from utils import predict

def histgradientboost(X_train, y_train):
    '''
    Final tuned HistGradientBoostingClassifier
    '''
    hgb_clf = ensemble.HistGradientBoostingClassifier(random_state=0, min_samples_leaf=5, learning_rate=0.5, max_depth=5, verbose=False)
    hgb_clf = hgb_clf.fit(X_train, y_train)

    hgb_pred = predict(hgb_clf)
    hgb_score = hgb_clf.score(X_train, y_train)

    return hgb_pred, hgb_score
